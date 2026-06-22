const textInput = document.getElementById('text-input');
const charCount = document.getElementById('char-count');
const analyzeBtn = document.getElementById('analyze-btn');
const resultBox = document.getElementById('result');
const errorBox = document.getElementById('error-box');
const sentimentTag = document.getElementById('sentiment-tag');
const confidencePct = document.getElementById('confidence-pct');
const barPositive = document.getElementById('bar-positive');
const barNegative = document.getElementById('bar-negative');
const valPositive = document.getElementById('val-positive');
const valNegative = document.getElementById('val-negative');
const logBox = document.getElementById('log');
const apiBaseInput = document.getElementById('api-base');

textInput.addEventListener('input', () => {
  charCount.textContent = textInput.value.length;
});

textInput.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
    analyzeBtn.click();
  }
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.add('visible');
  resultBox.classList.remove('visible');
}

function hideError() {
  errorBox.classList.remove('visible');
}

function renderResult(data) {
  const isPositive = data.label === 1;

  sentimentTag.textContent = data.sentiment;
  sentimentTag.className = 'sentiment-tag ' + (isPositive ? 'positive' : 'negative');

  const dominantProba = isPositive ? data.positive_proba : data.negative_proba;
  confidencePct.textContent = `確信度 ${(dominantProba * 100).toFixed(1)}%`;

  const posPct = (data.positive_proba * 100).toFixed(1);
  const negPct = (data.negative_proba * 100).toFixed(1);

  valPositive.textContent = `${posPct}%`;
  valNegative.textContent = `${negPct}%`;

  barPositive.style.width = '0%';
  barNegative.style.width = '0%';
  requestAnimationFrame(() => {
    barPositive.style.width = `${posPct}%`;
    barNegative.style.width = `${negPct}%`;
  });

  logBox.innerHTML =
    `<span class="key">label</span>: <span class="val">${data.label}</span>\n` +
    `<span class="key">positive_proba</span>: <span class="val">${data.positive_proba}</span>\n` +
    `<span class="key">negative_proba</span>: <span class="val">${data.negative_proba}</span>`;

  resultBox.classList.add('visible');
}

async function analyze() {
  const text = textInput.value.trim();
  hideError();

  if (!text) {
    showError('テキストを入力してください。');
    return;
  }

  const apiBase = apiBaseInput.value.trim().replace(/\/$/, '');
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = '分析中...';

  try {
    const response = await fetch(`${apiBase}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      const detail = errData?.detail ?? `サーバーエラー (status: ${response.status})`;
      throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    }

    const data = await response.json();
    renderResult(data);
  } catch (err) {
    showError(
      err instanceof TypeError
        ? 'APIに接続できませんでした。サーバーが起動しているか、URLを確認してください。'
        : `エラー: ${err.message}`
    );
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = '分析する';
  }
}

analyzeBtn.addEventListener('click', analyze);
