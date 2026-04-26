// ============================================
// DOM REFERENCES
// ============================================
const uploadZone      = document.getElementById('upload-zone');
const fileInput       = document.getElementById('file-input');
const fileNameEl      = document.getElementById('file-name');
const uploadBtn       = document.getElementById('upload-btn');
const uploadFeedback  = document.getElementById('upload-feedback');
const feedbackInner   = document.getElementById('upload-feedback-inner');

const searchInput     = document.getElementById('search-input');
const searchBtn       = document.getElementById('search-btn');

const resultsSection  = document.getElementById('results-section');
const resultsCount    = document.getElementById('results-count');
const resultsContainer = document.getElementById('results-container');

const loadingOverlay  = document.getElementById('loading-overlay');
const loadingText     = document.getElementById('loading-text');

const statusDot       = document.getElementById('status-dot');
const statusText      = document.getElementById('status-text');

let isUploaded = false;

// ============================================
// UPLOAD ZONE — click & drag-drop
// ============================================
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('dragover', e => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) {
    fileInput.files = e.dataTransfer.files;
    handleFileSelect(file);
  }
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFileSelect(fileInput.files[0]);
});

function handleFileSelect(file) {
  fileNameEl.textContent = file.name;
  fileNameEl.classList.add('selected');
  uploadBtn.disabled = false;
  hideFeedback();
}

// ============================================
// UPLOAD
// ============================================
uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) return;

  showLoading('Uploading & generating embeddings… (this may take a moment)');
  uploadBtn.disabled = true;

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res  = await fetch('/upload', { method: 'POST', body: formData });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Upload failed');

    isUploaded = true;
    showFeedback(`✅ ${data.message} — ${data.total_reviews} reviews indexed (text column: "${data.text_column}")`, 'success');
    enableSearch();
    updateStatus(data.total_reviews);
  } catch (err) {
    showFeedback(`❌ ${err.message}`, 'error');
    uploadBtn.disabled = false;
  } finally {
    hideLoading();
  }
});

// ============================================
// SEARCH
// ============================================
searchBtn.addEventListener('click', performSearch);
searchInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') performSearch();
});

async function performSearch() {
  const query = searchInput.value.trim();
  if (!query || !isUploaded) return;

  showLoading('Running semantic search…');

  try {
    const res  = await fetch('/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || 'Search failed');

    renderResults(data.results, data.total_found);
  } catch (err) {
    resultsSection.hidden = false;
    resultsCount.textContent = '';
    resultsContainer.innerHTML = renderEmpty('Something went wrong', err.message);
  } finally {
    hideLoading();
  }
}

// ============================================
// RENDER RESULTS
// ============================================
function renderResults(results, totalFound) {
  resultsSection.hidden = false;
  resultsSection.classList.add('fade-in');

  if (!results.length) {
    resultsCount.textContent = '';
    resultsContainer.innerHTML = renderEmpty('No results found', 'Try a different search term');
    return;
  }

  resultsCount.textContent = `Top ${totalFound} semantic match${totalFound !== 1 ? 'es' : ''}`;

  resultsContainer.innerHTML = results.map((r, i) => {
    const pct = (r.similarity_score * 100).toFixed(1);
    return `
    <div class="result-card" style="animation-delay:${i * .04}s">
      <div class="result-meta">
        ${r.product_name ? `
          <div class="product-name">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M6 2L3 6v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V6l-3-4z"/>
              <line x1="3" y1="6" x2="21" y2="6"/>
              <path d="M16 10a4 4 0 0 1-8 0"/>
            </svg>
            ${escapeHtml(r.product_name)}
          </div>` : ''}
        <span class="similarity-badge">${pct}% match</span>
      </div>
      <p class="review-text">${escapeHtml(r.review_text)}</p>
    </div>`;
  }).join('');

  // Scroll results into view
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderEmpty(title, hint) {
  return `
    <div class="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <line x1="21" y1="21" x2="16.65" y2="16.65"/>
        <line x1="8" y1="11" x2="14" y2="11"/>
      </svg>
      <p>${title}</p>
      <p class="hint">${hint}</p>
    </div>`;
}

// ============================================
// HELPERS
// ============================================
function enableSearch() {
  searchInput.disabled = false;
  searchBtn.disabled   = false;
  searchInput.focus();
}

function showLoading(text) {
  loadingText.textContent = text;
  loadingOverlay.hidden   = false;
}

function hideLoading() {
  loadingOverlay.hidden = true;
}

function showFeedback(msg, type) {
  uploadFeedback.hidden = false;
  feedbackInner.className = `feedback-inner ${type}`;
  feedbackInner.textContent = msg;
}

function hideFeedback() {
  uploadFeedback.hidden = true;
}

function updateStatus(count) {
  statusDot.classList.add('active');
  statusText.textContent = `${count} reviews loaded`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}
