'use strict';

/* ═══════════════════════════════════════════════════════════════════
   CropGuard — main.js
   Handles: file picking, drag-and-drop, AJAX prediction,
            skeleton loader, result rendering, toast notifications.
   ═══════════════════════════════════════════════════════════════════ */

/* ── DOM References ──────────────────────────────────────────────── */
const fileInput      = /** @type {HTMLInputElement}    */ (document.getElementById('leaf_image'));
const dropzone       = /** @type {HTMLLabelElement}    */ (document.getElementById('dropzone'));
const previewImg     = /** @type {HTMLImageElement}    */ (document.getElementById('preview-img'));
const emptyState     = /** @type {HTMLDivElement}      */ (document.getElementById('dropzone-empty'));
const diagnoseBtn    = /** @type {HTMLButtonElement}   */ (document.getElementById('diagnose-btn'));
const btnText        = /** @type {HTMLSpanElement}     */ (diagnoseBtn.querySelector('.btn-diagnose__text'));
const btnArrow       = /** @type {SVGElement}          */ (diagnoseBtn.querySelector('.btn-diagnose__arrow'));
const btnSpinner     = /** @type {SVGElement}          */ (diagnoseBtn.querySelector('.btn-spinner'));
const resetBtn       = /** @type {HTMLButtonElement}   */ (document.getElementById('reset-btn'));

const resultsSection = /** @type {HTMLElement}         */ (document.getElementById('results-section'));
const skeleton       = /** @type {HTMLDivElement}      */ (document.getElementById('skeleton'));
const resultsWrap    = /** @type {HTMLDivElement}      */ (document.getElementById('results-wrap'));
const resultsImg     = /** @type {HTMLImageElement}    */ (document.getElementById('results-img'));
const primaryCrop    = /** @type {HTMLHeadingElement}  */ (document.getElementById('primary-crop'));
const primaryDisease = /** @type {HTMLParagraphElement}*/ (document.getElementById('primary-disease'));
const confidenceRing = /** @type {HTMLDivElement}      */ (document.getElementById('confidence-ring'));
const confidencePct  = /** @type {HTMLSpanElement}     */ (document.getElementById('confidence-pct'));
const altList        = /** @type {HTMLDivElement}      */ (document.getElementById('alt-list'));
const toast          = /** @type {HTMLDivElement}      */ (document.getElementById('toast'));

/** Data URL of the currently loaded image — used for the results thumbnail */
let currentDataURL = null;
let toastTimer     = null;

/* ═══════════════════════════════════════════════════════════════════
   FILE HANDLING
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Load an image File into the dropzone preview and enable the button.
 * @param {File} file
 */
function loadFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    showToast('Please select a valid image (JPG, PNG, or WEBP).', true);
    return;
  }

  const reader = new FileReader();
  reader.onload = (e) => {
    currentDataURL = /** @type {string} */ (e.target.result);
    previewImg.src = currentDataURL;
    previewImg.hidden = false;
    emptyState.hidden = true;
    diagnoseBtn.disabled = false;
    // Clear any previous results when a new image is chosen
    hideResults();
  };
  reader.readAsDataURL(file);
}

/* ── File input change ───────────────────────────────────────────── */
fileInput.addEventListener('change', () => {
  if (fileInput.files?.[0]) loadFile(fileInput.files[0]);
});

/* ═══════════════════════════════════════════════════════════════════
   DRAG & DROP
   ═══════════════════════════════════════════════════════════════════ */
['dragenter', 'dragover', 'dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => e.preventDefault());
  document.body.addEventListener(evt, (e) => e.preventDefault());
});

dropzone.addEventListener('dragover',  () => dropzone.classList.add('dropzone--active'));
dropzone.addEventListener('dragleave', (e) => {
  // Only remove active state if leaving the dropzone entirely
  if (!dropzone.contains(/** @type {Node} */ (e.relatedTarget))) {
    dropzone.classList.remove('dropzone--active');
  }
});
dropzone.addEventListener('drop', (e) => {
  dropzone.classList.remove('dropzone--active');
  const file = e.dataTransfer?.files?.[0];
  if (file) {
    // Sync dropped file into the hidden file input (for FormData)
    try {
      const dt = new DataTransfer();
      dt.items.add(file);
      fileInput.files = dt.files;
    } catch (_) { /* Safari fallback — FormData will still use the file directly */ }
    loadFile(file);
  }
});

/* ═══════════════════════════════════════════════════════════════════
   PREDICTION (AJAX)
   ═══════════════════════════════════════════════════════════════════ */
diagnoseBtn.addEventListener('click', async () => {
  const file = fileInput.files?.[0];
  if (!file) {
    showToast('Please select or drop a leaf image first.', true);
    return;
  }

  setLoading(true);

  const formData = new FormData();
  formData.append('leaf_image', file);

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok && response.status !== 400 && response.status !== 500) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (data.error) {
      showToast(data.error, true);
      setLoading(false);
      return;
    }

    renderResults(data.results);

  } catch (err) {
    console.error('Prediction error:', err);
    showToast('Network error — make sure the Flask server is running.', true);
    setLoading(false);
  }
});

/* ═══════════════════════════════════════════════════════════════════
   LOADING STATE
   ═══════════════════════════════════════════════════════════════════ */
/**
 * Toggle the button and skeleton loader.
 * @param {boolean} isLoading
 */
function setLoading(isLoading) {
  diagnoseBtn.disabled = isLoading;
  btnText.textContent  = isLoading ? 'Analyzing…' : 'Analyze Leaf';
  btnArrow.hidden      = isLoading;
  btnSpinner.hidden    = !isLoading;

  if (isLoading) {
    resultsSection.hidden = false;
    skeleton.hidden       = false;
    resultsWrap.hidden    = true;
    // Smooth scroll to results area
    setTimeout(() => {
      resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }, 50);
  }
}

/* ═══════════════════════════════════════════════════════════════════
   RENDER RESULTS
   ═══════════════════════════════════════════════════════════════════ */
/**
 * Populate and reveal the results panel.
 * @param {Array<{label: string, confidence: number}>} results
 */
function renderResults(results) {
  if (!results?.length) {
    showToast('No predictions returned.', true);
    setLoading(false);
    return;
  }

  /* ── Thumbnail ── */
  if (currentDataURL) resultsImg.src = currentDataURL;

  /* ── Primary result ── */
  const top    = results[0];
  const parts  = top.label.split(' — ');
  primaryCrop.textContent    = parts[0] ?? top.label;
  primaryDisease.textContent = parts[1] ?? '';

  /* ── Animate confidence ring (conic-gradient + counter) ── */
  const targetPct = Math.round(top.confidence);
  animateValue(0, targetPct, 1100, (v) => {
    confidenceRing.style.setProperty('--pct', String(v));
    confidencePct.textContent = `${v}%`;
  });

  /* ── Alternative results ── */
  altList.innerHTML = '';
  const alts = results.slice(1, 4); // Show up to 3 alternatives
  alts.forEach((r, i) => {
    const item = document.createElement('div');
    item.className = 'alt-item';
    item.setAttribute('role', 'listitem');
    item.style.opacity = '0';
    item.style.transform = 'translateY(8px)';
    item.style.transition = `opacity 0.4s ease ${i * 80}ms, transform 0.4s ease ${i * 80}ms`;

    item.innerHTML = `
      <div>
        <p class="alt-item__name">${escapeHTML(r.label)}</p>
        <div class="alt-bar" role="progressbar" aria-valuenow="${r.confidence.toFixed(1)}" aria-valuemin="0" aria-valuemax="100">
          <div class="alt-bar__fill" data-pct="${r.confidence.toFixed(2)}"></div>
        </div>
      </div>
      <span class="alt-pct">${r.confidence.toFixed(1)}%</span>
    `;
    altList.appendChild(item);
  });

  /* ── Transition: skeleton → results ── */
  skeleton.hidden    = true;
  resultsWrap.hidden = false;
  setLoading(false);
  diagnoseBtn.disabled = false;

  /* ── Staggered animation for alt items ── */
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      // Fade in alt items
      altList.querySelectorAll('.alt-item').forEach((el) => {
        el.style.opacity   = '1';
        el.style.transform = 'translateY(0)';
      });
      // Animate confidence bars (double rAF ensures CSS transition fires)
      altList.querySelectorAll('.alt-bar__fill').forEach((bar) => {
        bar.style.width = `${bar.dataset.pct}%`;
      });
    });
  });

  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/* ── Hide results (called when new image is chosen) ─────────────── */
function hideResults() {
  resultsSection.hidden = true;
  skeleton.hidden       = true;
  resultsWrap.hidden    = true;
}

/* ── Reset button ────────────────────────────────────────────────── */
resetBtn?.addEventListener('click', () => {
  // Clear file input
  fileInput.value     = '';
  currentDataURL      = null;
  previewImg.src      = '';
  previewImg.hidden   = true;
  emptyState.hidden   = false;
  diagnoseBtn.disabled = true;
  hideResults();
  window.scrollTo({ top: 0, behavior: 'smooth' });
});

/* ═══════════════════════════════════════════════════════════════════
   UTILITIES
   ═══════════════════════════════════════════════════════════════════ */

/**
 * Animate a numeric value from `from` to `to` over `duration` ms.
 * Calls `onUpdate` each frame with the current integer value.
 * @param {number}   from
 * @param {number}   to
 * @param {number}   duration  ms
 * @param {(v: number) => void} onUpdate
 */
function animateValue(from, to, duration, onUpdate) {
  const startTime = performance.now();
  const tick = (now) => {
    const progress = Math.min((now - startTime) / duration, 1);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - progress, 3);
    const value  = Math.round(from + (to - from) * eased);
    onUpdate(value);
    if (progress < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

/**
 * Show a toast notification.
 * @param {string}  message
 * @param {boolean} [isError=false]
 */
function showToast(message, isError = false) {
  clearTimeout(toastTimer);
  toast.textContent = message;
  toast.className   = `toast toast--show${isError ? ' toast--error' : ''}`;
  toastTimer = setTimeout(() => {
    toast.classList.remove('toast--show', 'toast--error');
  }, 3800);
}

/**
 * Escape HTML special characters to prevent XSS from server data.
 * @param {string} str
 * @returns {string}
 */
function escapeHTML(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}
