const FEATURES = ["relative_humidity","cloud_cover","dew_point","surface_pressure","temp_mean",
  "temp_max","temp_min","wind_speed_mean","wind_speed_max","wind_gust","wind_direction",
  "rain_lag1","day_of_year"];

let IS_LOCAL = true;

// Real HTML escaping. The previous version "replaced" & with & and < with <,
// i.e. it escaped nothing and returned its input untouched.
function esc(s){
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}

// A metric we do not actually have must render as "-", never as NaN or a crash.
function num(v, digits){
  return (typeof v === "number" && isFinite(v)) ? v.toFixed(digits === undefined ? 3 : digits) : "&mdash;";
}

function liveHeuristic(f){
  let score = 0, why = [];
  if (f.relative_humidity >= 80){ score++; why.push("humidity ≥80%"); }
  if (f.cloud_cover >= 60){ score++; why.push("cloud cover ≥60%"); }
  if (f.rain_lag1 > 20){ score++; why.push("yesterday's rain >20 mm"); }
  if (f.wind_speed_mean >= 25){ score++; why.push("strong wind ≥25 km/h"); }
  return { score, why };
}

function applyResult(d){
  if (typeof d.heavy_rain_threshold_mm === "number") window.HEAVY_MM = d.heavy_rain_threshold_mm;
  document.getElementById("risk").textContent = d.risk_level + " RISK";
  document.getElementById("risk").className = "risk " + d.risk_level;
  document.getElementById("prob").innerHTML = (d.probability*100).toFixed(0) + "<small>%</small>";
  document.getElementById("probBar").style.width = (d.probability*100).toFixed(0) + "%";
  document.getElementById("rel").textContent = "RELIABILITY " + d.reliability.reliability;
  document.getElementById("rel").className = "rel " + d.reliability.reliability;
  document.getElementById("relReasons").innerHTML = d.reliability.reasons.map(r => "<li>• " + esc(r) + "</li>").join("");
  document.getElementById("disc").textContent = d.disclaimer;
  drawShap(d.explanation);
  if (document.getElementById("status")) {
    document.getElementById("status").textContent = "Ensemble agreement (std): " + d.model_agreement_std + " · reliability OOD score: " + d.reliability.ood_score + " · OOD flag: " + d.reliability.ood_flag;
  }
}

function drawShap(exps){
  const container = document.getElementById("shapChart");
  if (!container) return;
  const max = Math.max(...exps.map(e => Math.abs(e.shap)), 0.001);
  let html = "";
  exps.forEach(e => {
    const w = Math.abs(e.shap)/max*100;
    const up = e.shap >= 0;
    html += `<div class="shap-bar">
      <div class="label">
        <span>${esc(e.feature)} <span style="color:var(--mut)">(${e.value})</span></span>
        <span class="${up?'up':'down'}">${up?'+':''}${e.shap.toFixed(3)}</span></div>
      <div class="bar"><div style="width:${w.toFixed(0)}%;background:${up?'var(--red)':'var(--primary)'}"></div></div></div>`;
  });
  container.innerHTML = html;
  const note = document.getElementById("shapNote");
  if (note) note.textContent = "Red pushes towards heavy rain, blue pushes away. Contributions are true SHAP values from the model.";
}

async function loadMetrics(){
  let d;
  try {
    const r = await fetch("/metrics");
    if (!r.ok) throw new Error();
    d = await r.json();
  } catch(e){
    IS_LOCAL = false;
    const tag = document.getElementById("modeTag");
    if (tag) tag.textContent = "HOSTED DEMO MODE";
    d = await (await fetch("/frontend/demo_metrics.json")).json();
  }
  const m = d.metrics.test;
  // 'roc_auc' used to hold balanced accuracy mislabelled; it is now null until
  // the model is retrained. Render what we can prove, dash the rest.
  const rows = [["Balanced accuracy", m.balanced_accuracy], ["Recall (POD)", m.recall],
    ["Precision", m.precision], ["F1-score", m.f1], ["CSI", m.csi],
    ["FAR", m.far], ["Specificity", m.specificity], ["Accuracy", m.accuracy],
    ["ROC-AUC", m.roc_auc], ["PR-AUC", m.pr_auc], ["Brier", m.brier]];
  const perfTable = document.getElementById("perfTable");
  if (perfTable) {
    perfTable.innerHTML = rows.map(r =>
      `<tr><td>${r[0]}</td><td class="num">${num(r[1])}</td></tr>`).join("")
      + `<tr><td>Samples</td><td class="num">${m.n.toLocaleString()} (${m.n_heavy.toLocaleString()} heavy)</td></tr>`;
  }
  const perfNote = document.getElementById("perfNote");
  if (perfNote) {
    perfNote.innerHTML = "Balanced accuracy = mean of recall and specificity; on " +
      (100 * m.n_heavy / m.n).toFixed(1) + "% positives, plain accuracy is dominated by " +
      "the majority class. ROC-AUC, PR-AUC and Brier need saved probabilities and are " +
      "pending a retrain (&mdash; means not yet measured, not zero).";
  }
  const cm = m.confusion_matrix;
  const cmTable = document.getElementById("cmTable");
  if (cmTable) {
    cmTable.innerHTML =
      `<tr><th></th><th>Pred: no</th><th>Pred: heavy</th></tr>
       <tr><td>Actual: no</td><td class="num">${cm.tn.toLocaleString()}</td><td class="num">${cm.fp.toLocaleString()}</td></tr>
       <tr><td>Actual: heavy</td><td class="num">${cm.fn.toLocaleString()}</td><td class="num">${cm.tp.toLocaleString()}</td></tr>`;
  }
  const gi = d.global_shap.importance;
  const gmax = Math.max(...gi.map(i => i.importance));
  const globalChart = document.getElementById("globalChart");
  if (globalChart) {
    globalChart.innerHTML = gi.slice(0,10).map(i =>
      `<div class="global-item">
        <span>${esc(i.feature)}</span>
        <div class="bar"><div style="width:${i.importance/gmax*100}%;background:var(--primary)"></div></div>
        <span>${i.importance.toFixed(3)}</span></div>`).join("");
  }
}

async function applyDemoCase(){
  const d = (await (await fetch("/frontend/demo_cases.json")).json())["heavy_rain_day"];
  const predLabel = document.getElementById("predLabel");
  if (predLabel) predLabel.textContent = "Prediction (precomputed): " + d.prediction + " · heavy-rain threshold >" + d.heavy_rain_threshold_mm + " mm/day";
  applyResult(d);
}

function initTheme(){
  const saved = localStorage.getItem("theme");
  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
  updateThemeIcon(theme);
}
function toggleTheme(){
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("theme", next);
  updateThemeIcon(next);
}
function updateThemeIcon(theme){
  const btn = document.getElementById("themeToggle");
  if (btn) btn.textContent = theme === "dark" ? "☀️ Light" : "🌙 Dark";
}
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  document.getElementById("themeToggle")?.addEventListener("click", toggleTheme);
  loadMetrics();
  if (document.getElementById("predLabel")) {
    setTimeout(() => { if (!IS_LOCAL) applyDemoCase(); }, 500);
  }
});

window.FEATURES = FEATURES;
// A plain assignment copied the initial `true` and never updated, so anything
// reading window.IS_LOCAL saw "local" forever. A getter always reports the truth.
Object.defineProperty(window, "IS_LOCAL", { get: () => IS_LOCAL });
Object.defineProperty(window, "esc", { value: esc });
Object.defineProperty(window, "num", { value: num });
window.liveHeuristic = liveHeuristic;
window.applyResult = applyResult;
window.drawShap = drawShap;
window.loadMetrics = loadMetrics;
window.applyDemoCase = applyDemoCase;