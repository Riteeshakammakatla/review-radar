from flask import Flask, render_template, request, jsonify
from fastembed import TextEmbedding
import pandas as pd
import numpy as np
import faiss
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------------------------------------------------------
# Load model at startup — Railway has enough RAM for this
# ---------------------------------------------------------------------------
print("[INFO] Loading embedding model...")
model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
print("[OK] Model loaded!")

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------
review_texts = []
product_names = []
faiss_index = None

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
    embeddings = np.array(list(model.embed(texts))).astype('float32')
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
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
    global review_texts, product_names, faiss_index
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath, sep=None, engine='python', on_bad_lines='warn')

        text_col = _find_column(df.columns, REVIEW_COL_CANDIDATES)
        if text_col is None:
            text_col = _auto_detect_text_column(df)
        if text_col is None:
            return jsonify({'error': 'Could not detect text column'}), 400

        prod_col = _find_column(df.columns, PRODUCT_COL_CANDIDATES)
        review_texts = df[text_col].fillna('').astype(str).tolist()
        product_names = (
            df[prod_col].fillna('').astype(str).tolist()
            if prod_col else [None] * len(review_texts)
        )

        print(f"[INFO] Indexing {len(review_texts)} reviews...")
        faiss_index = _build_index(review_texts)
        print("[OK] FAISS index ready!")

        return jsonify({
            'message': f'File uploaded & indexed successfully',
            'total_reviews': len(review_texts),
            'text_column': text_col
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    if not query or faiss_index is None:
        return jsonify({'error': 'Please upload a file first'}), 400

    query_embedding = np.array(list(model.embed([query]))).astype('float32')
    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_index.search(query_embedding, 10)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            'review_text': review_texts[idx],
            'product_name': product_names[idx],
            'similarity_score': round(float(score), 4)
        })
    return jsonify({'results': results, 'total_found': len(results)})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
