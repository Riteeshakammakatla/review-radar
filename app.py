from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# Load the sentence-transformer model once at startup
# all-MiniLM-L6-v2 produces 384-dimensional embeddings, is fast & lightweight
# ---------------------------------------------------------------------------
print("[INFO] Loading sentence-transformer model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("[OK] Model loaded successfully!")

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
reviews_data = []          # list[dict]  – raw review rows
review_texts = []          # list[str]   – the text column used for embeddings
product_names = []         # list[str|None]
faiss_index = None         # faiss.IndexFlatIP (inner-product / cosine)

# Column name candidates (case-insensitive)
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
    """Flexible column matching ignoring case, spaces, and underscores."""
    norm_candidates = [_normalize(c) for c in candidates]
    for col in df_columns:
        if _normalize(col) in norm_candidates:
            return col
    return None


def _auto_detect_text_column(df):
    """Fallback: pick the column with the longest average string length."""
    best_col = None
    best_avg = 0
    for col in df.columns:
        if df[col].dtype == object:
            avg_len = df[col].dropna().astype(str).str.len().mean()
            if avg_len > best_avg:
                best_avg = avg_len
                best_col = col
    return best_col


def _build_index(texts):
    """
    Generate embeddings for all review texts and build a FAISS index.

    Uses IndexFlatIP (inner product) on L2-normalised vectors,
    which is equivalent to cosine similarity — ideal for semantic search.

    For 200k reviews the encode step dominates; FAISS flat search is ~ms.
    """
    # Encode in batches (sentence-transformers handles this internally)
    # normalize_embeddings=True  →  cosine similarity via inner product
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype('float32')

    dim = embeddings.shape[1]  # 384 for MiniLM
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global reviews_data, review_texts, product_names, faiss_index

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # sep=None + engine='python' tells pandas to guess the separator (comma, tab, etc.)
        df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='warn')
        reviews_data = df.to_dict(orient='records')

        # --- Detect the review-text column ---
        text_col = _find_column(df.columns, REVIEW_COL_CANDIDATES)
        if text_col is None:
            text_col = _auto_detect_text_column(df)
        if text_col is None:
            return jsonify({'error': 'Could not detect a review text column'}), 400

        # --- Detect the product-name column (optional) ---
        prod_col = _find_column(df.columns, PRODUCT_COL_CANDIDATES)

        # --- Prepare clean text lists ---
        review_texts = df[text_col].fillna('').astype(str).tolist()
        product_names = (
            df[prod_col].fillna('').astype(str).tolist()
            if prod_col else [None] * len(review_texts)
        )

        # --- Build FAISS index ---
        print(f"[INFO] Building FAISS index for {len(review_texts)} reviews "
              f"(text column: '{text_col}')...")
        faiss_index = _build_index(review_texts)
        print("[OK] FAISS index ready!")

        return jsonify({
            'message': 'File uploaded & embeddings indexed successfully',
            'total_reviews': len(review_texts),
            'columns': list(df.columns),
            'text_column': text_col,
            'product_column': prod_col,
        })

    except Exception as e:
        # Reset state on failure
        reviews_data, review_texts, product_names, faiss_index = [], [], [], None
        return jsonify({'error': str(e)}), 500


@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    top_k = min(int(data.get('top_k', 10)), 50)   # allow frontend override

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    if faiss_index is None or not review_texts:
        return jsonify({'error': 'No data uploaded yet'}), 400

    # --- Encode the query ---
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype('float32')

    # --- FAISS similarity search ---
    scores, indices = faiss_index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue  # fewer results than top_k
        results.append({
            'review_text': review_texts[idx],
            'product_name': product_names[idx] or None,
            'similarity_score': round(float(score), 4),
        })

    return jsonify({
        'results': results,
        'total_found': len(results),
    })


if __name__ == '__main__':
    # Render provides the port in an environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
