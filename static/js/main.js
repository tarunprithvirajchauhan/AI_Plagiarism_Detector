/* ==========================================================================
   VeriText AI - Main JavaScript Interactivity & API Integration
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  initTabs();
  initTextCounter();
});

/* ==================== 1. Theme Toggler ==================== */
function initTheme() {
  const toggleBtn = document.getElementById("themeToggleBtn");
  const themeIcon = document.getElementById("themeIcon");
  const themeLabel = document.getElementById("themeLabel");
  const html = document.documentElement;

  // Check saved theme or default to dark
  const savedTheme = localStorage.getItem("veritext_theme") || "dark";
  setTheme(savedTheme);

  toggleBtn.addEventListener("click", () => {
    const currentTheme = html.getAttribute("data-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    setTheme(newTheme);
  });

  function setTheme(theme) {
    html.setAttribute("data-theme", theme);
    localStorage.setItem("veritext_theme", theme);
    if (theme === "dark") {
      themeIcon.className = "fa-solid fa-sun";
      themeLabel.textContent = "Light Mode";
    } else {
      themeIcon.className = "fa-solid fa-moon";
      themeLabel.textContent = "Dark Mode";
    }
  }
}

/* ==================== 2. Navigation Tabs ==================== */
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabBtns.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));

      btn.classList.add("active");
      document.getElementById(targetId).classList.add("active");
    });
  });
}

/* ==================== 3. Real-Time Text Counter ==================== */
function initTextCounter() {
  const textInput = document.getElementById("textInput");
  const wordCount = document.getElementById("wordCount");
  const charCount = document.getElementById("charCount");
  const readingTime = document.getElementById("readingTime");

  textInput.addEventListener("input", () => {
    const text = textInput.value;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    const chars = text.length;
    const readMin = (words / 200).toFixed(1);

    wordCount.textContent = words.toLocaleString();
    charCount.textContent = chars.toLocaleString();
    readingTime.textContent = readMin;
  });
}

/* ==================== 4. Preset Samples ==================== */
const SAMPLES = {
  human_essay: `Cars have been around since Henry Ford created and built the first Model T. They played a major role in our everyday lives since then. But now, people are starting to question if limiting car usage would be a good thing. To me, limiting the use of cars might be a good thing to do. In German Suburb, Life Goes On Without Cars by Elizabeth Rosenthal states how automobiles are the linchpin of suburbs where middle class families tend to make their homes.`,
  
  ai_essay: `Artificial intelligence has profoundly transformed modern society through rapid advancements in natural language processing and automated data synthesis. Generative AI architectures leverage vast neural network datasets to compose cohesive, highly structured essays. Furthermore, machine learning models streamline workflows across healthcare, software development, and academic research.`,
  
  tech_paper: `In this paper, we evaluate the performance of TF-IDF feature selection combined with Complement Naive Bayes classifiers. The empirical evaluation demonstrates that hyperparameter tuning of alpha parameter yields an optimal F1-score of 0.9703 on the 487k essay benchmark dataset.`
};

function loadSample(key) {
  const textInput = document.getElementById("textInput");
  textInput.value = SAMPLES[key] || "";
  textInput.dispatchEvent(new Event("input"));
  analyzeText();
}

function clearText() {
  const textInput = document.getElementById("textInput");
  textInput.value = "";
  textInput.dispatchEvent(new Event("input"));
  document.getElementById("resultPanel").innerHTML = `
    <div class="empty-state">
      <i class="fa-solid fa-microscope"></i>
      <p>Enter text on the left and click <strong>Analyze Text</strong> to view real-time AI detection metrics.</p>
    </div>
  `;
}

/* ==================== 5. Analyze Text API ==================== */
async function analyzeText() {
  const textInput = document.getElementById("textInput");
  const resultPanel = document.getElementById("resultPanel");
  const text = textInput.value.trim();

  if (!text) {
    alert("Please enter some text before analyzing.");
    return;
  }

  // Show loading spinner
  resultPanel.innerHTML = `
    <div class="spinner"></div>
    <p style="color: var(--text-secondary); margin-top: 1rem;">Analyzing text patterns with Naive Bayes...</p>
  `;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    });

    const data = await response.json();

    if (!response.ok) {
      resultPanel.innerHTML = `<p style="color: var(--ai-color);">${data.error || "Error analyzing text."}</p>`;
      return;
    }

    renderResults(data);
  } catch (err) {
    console.error(err);
    resultPanel.innerHTML = `<p style="color: var(--ai-color);">Network error connecting to Flask server.</p>`;
  }
}

function renderResults(data) {
  const resultPanel = document.getElementById("resultPanel");
  const isAI = data.is_ai;
  const badgeClass = isAI ? "badge-ai" : "badge-human";
  const badgeIcon = isAI ? "fa-solid fa-robot" : "fa-solid fa-user-pen";
  const fillClass = isAI ? "fill-ai" : "fill-human";
  const conf = data.confidence;

  let sentencesHtml = "";
  if (data.sentence_breakdown && data.sentence_breakdown.length > 0) {
    sentencesHtml = `
      <div class="sentence-breakdown">
        <h4><i class="fa-solid fa-align-left"></i> Sentence Breakdown & Highlights</h4>
        ${data.sentence_breakdown.map(s => `
          <div class="sentence-item ${s.label === 'AI' ? 'sentence-ai' : 'sentence-human'}">
            <strong>[${s.label} ${s.confidence}%]</strong> ${escapeHtml(s.sentence)}
          </div>
        `).join('')}
      </div>
    `;
  }

  resultPanel.innerHTML = `
    <div class="gauge-box">
      <div class="result-badge ${badgeClass}">
        <i class="${badgeIcon}"></i> ${data.label}
      </div>
      
      <div style="font-size: 2.2rem; font-weight: 800; color: ${isAI ? 'var(--ai-color)' : 'var(--human-color)'};">
        ${conf}% <span style="font-size: 1.1rem; font-weight: 600; color: var(--text-secondary);">Confidence</span>
      </div>

      <div class="confidence-bar-container">
        <div class="confidence-bar-fill ${fillClass}" style="width: ${conf}%;"></div>
      </div>

      <div class="probability-labels">
        <span><i class="fa-solid fa-user"></i> Human: ${data.human_probability}%</span>
        <span><i class="fa-solid fa-robot"></i> AI: ${data.ai_probability}%</span>
      </div>
    </div>

    <div style="width: 100%; font-size: 0.85rem; color: var(--text-muted); border-top: 1px solid var(--border-color); padding-top: 0.8rem; margin-top: 0.5rem; display: flex; justify-content: space-between;">
      <span>Latency: <strong>${data.latency_ms} ms</strong></span>
      <span>Words analyzed: <strong>${data.stats.words}</strong></span>
    </div>

    ${sentencesHtml}
  `;
}

/* ==================== 6. Batch Processing ==================== */
async function runBatchAnalysis() {
  const input = document.getElementById("batchInput").value.trim();
  const container = document.getElementById("batchResultsContainer");

  if (!input) {
    alert("Please enter text items separated by double line breaks.");
    return;
  }

  const items = input.split(/\n\s*\n/).map((t, i) => ({ id: `Item ${i+1}`, text: t.trim() })).filter(item => item.text.length > 0);

  container.innerHTML = `<div class="spinner"></div><p style="text-align: center;">Processing ${items.length} items...</p>`;

  try {
    const res = await fetch("/api/batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: items })
    });
    const data = await res.json();

    if (!res.ok) {
      container.innerHTML = `<p style="color: var(--ai-color);">${data.error}</p>`;
      return;
    }

    container.innerHTML = `
      <h3>Batch Analysis Complete (${data.total_processed} items in ${data.latency_ms} ms)</h3>
      <table class="metrics-table" style="margin-top: 1rem;">
        <thead>
          <tr>
            <th>Item ID</th>
            <th>Snippet</th>
            <th>Prediction</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>
          ${data.results.map(r => `
            <tr>
              <td><strong>${r.id}</strong></td>
              <td>${escapeHtml(r.text_snippet)}</td>
              <td><span class="${r.is_ai ? 'badge-ai' : 'badge-human'}" style="padding: 0.2rem 0.6rem; font-size: 0.85rem;">${r.label}</span></td>
              <td><strong>${r.confidence}%</strong></td>
            </tr>
          `).join('')}
        </tbody>
      </table>
    `;
  } catch (err) {
    console.error(err);
    container.innerHTML = `<p style="color: var(--ai-color);">Error executing batch prediction.</p>`;
  }
}

function escapeHtml(text) {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return text.replace(/[&<>"']/g, m => map[m]);
}
