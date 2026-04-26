from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import faiss
import logging
import os
import time
import gc

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy model loading — only loads when first needed, not at server boot.
# This keeps startup fast and avoids Railway killing the process during init.
# ---------------------------------------------------------------------------
_model = None

def get_model():
    global _model
    if _model is None:
        logger.info("Loading fastembed model (BAAI/bge-small-en-v1.5)...")
        from fastembed import TextEmbedding
        # Limit threads to 1 or 2 to drastically reduce memory usage during embedding
        _model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
        logger.info("Model loaded successfully!")
    return _model

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
review_texts = []
product_names = []
faiss_index = None

# Set to 25000 to handle the entire dataset for maximum search accuracy
MAX_ROWS = 25000

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------
REVIEW_COL_CANDIDATES = [
    'review', 'text', 'review text', 'review_text', 'comment', 'body',
    'content', 'reviews text', 'reviews_text', 'review body', 'review_body',
    'feedback', 'description', 'summary', 'review content'
]
PRODUCT_COL_CANDIDATES = [
    'product', 'product name', 'product_name', 'name', 'item', 'title',
    'product title', 'brand'
]


def _normalize(s):
    return "".join(s.lower().split()).replace("_", "")


def _find_column(df_columns, candidates):
    norm_candidates = [_normalize(c) for c in candidates]
    for col in df_columns:
        if _normalize(col) in norm_candidates:
            return col
    return None


def _auto_detect_text_column(df):
    best_col, best_avg = None, 0
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > best_avg:
                best_avg = avg_len
                best_col = col
    return best_col


# ---------------------------------------------------------------------------
# Text preprocessing — cleans review text before embedding
# ---------------------------------------------------------------------------
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    # Strip whitespace, collapse multiple spaces
    text = " ".join(text.split())
    return text


# ---------------------------------------------------------------------------
# FAISS index builder — processes in batches for memory efficiency
# ---------------------------------------------------------------------------
def _build_index(texts):
    model = get_model()
    # Reduce batch size to prevent memory spikes
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        # model.embed() is for DOCUMENTS (no prefix added)
        batch_emb = np.array(list(model.embed(batch))).astype('float32')
        all_embeddings.append(batch_emb)
        logger.info(f"  Embedded batch {i // batch_size + 1}: "
                     f"{min(i + batch_size, len(texts))}/{len(texts)} reviews")

    embeddings = np.vstack(all_embeddings)

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = cosine sim on normalized vectors
    index.add(embeddings)

    logger.info(f"FAISS index built: {index.ntotal} vectors, {dim} dimensions")
    return index


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204


@app.route('/upload', methods=['POST'])
def upload():
    global review_texts, product_names, faiss_index

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    logger.info(f"File saved: {filepath}")

    try:
        start_time = time.time()

        # Read only the rows we need — avoids loading entire 21k-row CSV into RAM
        try:
            df = pd.read_csv(filepath, nrows=MAX_ROWS)
        except Exception:
            df = pd.read_csv(filepath, sep=None, engine='python',
                             on_bad_lines='warn', nrows=MAX_ROWS)

        logger.info(f"CSV loaded: {len(df)} rows, columns: {list(df.columns)}")

        # Detect columns
        text_col = _find_column(df.columns, REVIEW_COL_CANDIDATES)
        if text_col is None:
            text_col = _auto_detect_text_column(df)
        if text_col is None:
            return jsonify({'error': 'Could not detect a review text column. '
                           f'Available columns: {list(df.columns)}'}), 400

        prod_col = _find_column(df.columns, PRODUCT_COL_CANDIDATES)
        logger.info(f"Using text column: '{text_col}', product column: '{prod_col}'")

        # Preprocess texts
        review_texts = [preprocess_text(t) for t in df[text_col].fillna('').astype(str)]
        product_names = (
            df[prod_col].fillna('').astype(str).tolist()
            if prod_col else [None] * len(review_texts)
        )

        # Filter out empty reviews
        valid_indices = [i for i, t in enumerate(review_texts) if len(t) > 5]
        review_texts = [review_texts[i] for i in valid_indices]
        product_names = [product_names[i] for i in valid_indices]

        logger.info(f"After filtering: {len(review_texts)} valid reviews")

        # Build FAISS index
        faiss_index = _build_index(review_texts)

        # Clean up to free memory
        del df
        gc.collect()

        elapsed = round(time.time() - start_time, 1)
        logger.info(f"Upload complete in {elapsed}s")

        return jsonify({
            'message': f'Indexed {len(review_texts)} reviews in {elapsed}s',
            'total_reviews': len(review_texts),
            'text_column': text_col,
            'product_column': prod_col,
        })

    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        review_texts, product_names, faiss_index = [], [], None
        gc.collect()
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    top_k = min(int(data.get('top_k', 10)), 50)

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
    if faiss_index is None or not review_texts:
        return jsonify({'error': 'Please upload a CSV file first'}), 400

    model = get_model()

    # =========================================================================
    # CRITICAL FIX: Use query_embed() NOT embed() for search queries!
    #
    # BGE models (like bge-small-en-v1.5) require a special prefix for queries
    # to distinguish them from documents. fastembed's query_embed() handles
    # this automatically. Using embed() for queries produces POOR results
    # because the model treats the query as a document instead.
    # =========================================================================
    query_embedding = np.array(
        list(model.query_embed([preprocess_text(query)]))
    ).astype('float32')
    faiss.normalize_L2(query_embedding)

    # FAISS search — returns cosine similarity scores (higher = more relevant)
    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            'review_text': review_texts[idx],
            'product_name': product_names[idx] or None,
            'similarity_score': round(float(score), 4),
        })

    logger.info(f"Search '{query}' -> {len(results)} results "
                f"(top score: {results[0]['similarity_score'] if results else 'N/A'})")

    return jsonify({'results': results, 'total_found': len(results)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
