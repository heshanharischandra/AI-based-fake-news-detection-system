const state = { file: null, previewUrl: null, jobId: null, timer: null, busy: false, validating: false, selectionId: 0 };
const $ = (selector) => document.querySelector(selector);

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", "'":"&#39;", '"':"&quot;"}[character]));
}

function setNotice(message = "", kind = "error") {
  const notice = $("#setup-notice");
  notice.textContent = message;
  notice.classList.toggle("success", Boolean(message) && kind === "success");
  notice.classList.toggle("error", Boolean(message) && kind === "error");
  $("#dropzone").setAttribute("aria-invalid", String(Boolean(message) && kind === "error"));
}
const checkDefaults = {format: "PNG, JPG or JPEG", size: "Up to 12 MB", content: "Checked after selection", dimensions: "Up to 16 megapixels"};
function setUploadCheck(key, status = "waiting", detail = checkDefaults[key]) {
  const check = $(`#check-${key}`);
  check.dataset.state = status;
  check.querySelector(".check-icon").textContent = status === "passed" ? "✓" : status === "error" ? "!" : "○";
  check.querySelector("small").textContent = detail;
}
function updateUploadControls() {
  $("#analyze-button").disabled = state.busy || state.validating || !state.file;
  $("#analyze-label").textContent = state.busy ? "Verification in progress…" : state.validating ? "Checking your image…" : state.file ? "Run verification" : "Select an image to begin";
  $("#dropzone").setAttribute("aria-disabled", String(state.busy));
  $("#dropzone").setAttribute("aria-busy", String(state.validating));
  $("#file-input").disabled = state.busy;
  $("#remove-file").disabled = state.busy;
  $("#reset-button").disabled = state.busy;
  $("#dropzone-title").textContent = state.file ? "Choose a different image" : "Drop a news image here";
  $("#preview-status").textContent = state.validating ? "Checking…" : state.file ? "Ready to review" : "No image";
  $("#preview-status").classList.toggle("ready", Boolean(state.file));
}
function hasNumber(value) { return value !== null && value !== undefined && value !== "" && Number.isFinite(Number(value)); }
function formatPercent(value, digits = 2) {
  if (!hasNumber(value)) return "—";
  const number = Number(value);
  return `${(Math.abs(number) > 1 ? number : number * 100).toFixed(digits)}%`;
}
function formatNumber(value, digits = 2) { return hasNumber(value) ? Number(value).toFixed(digits) : "—"; }
function cleanStatus(value) { return String(value || "pending").replaceAll("_", " "); }
function badge(status, kind = "") { return `<span class="badge ${kind}">${escapeHtml(cleanStatus(status))}</span>`; }
function metric(label, value) { return `<div class="metric"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`; }
function rows(items) { return `<div class="detail-list">${items.filter((item) => item[1] !== undefined && item[1] !== null && item[1] !== "").map(([label, value]) => `<div class="detail-row"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join("")}</div>`; }

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join(". ") : data.detail;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return data;
}

function showPreview(source) {
  const image = $("#preview-image");
  if (state.previewUrl && state.previewUrl.startsWith("blob:")) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = source;
  image.src = source;
  image.hidden = false;
  $("#preview-empty").classList.add("hidden");
}

function resetSelection() {
  state.selectionId += 1;
  state.validating = false;
  state.file = null;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $("#file-input").value = "";
  $("#preview-image").hidden = true;
  $("#preview-image").removeAttribute("src");
  $("#preview-empty").classList.remove("hidden");
  $("#selected-file").classList.add("hidden");
  $("#file-name").textContent = "";
  $("#file-meta").textContent = "";
  Object.keys(checkDefaults).forEach((key) => setUploadCheck(key));
  updateUploadControls();
}

async function selectFile(file) {
  if (!file || state.busy) return;
  resetAnalysis();
  const selectionId = state.selectionId;
  state.validating = true;
  updateUploadControls();
  setNotice("");
  let previewUrl = null;
  let currentCheck = "format";
  try {
    const extension = file.name.split(".").pop().toLowerCase();
    if (!file.name.includes(".") || !["png", "jpg", "jpeg"].includes(extension)) {
      throw new Error("Choose a PNG, JPG or JPEG image only.");
    }
    setUploadCheck("format", "passed", extension.toUpperCase());
    currentCheck = "size";
    if (!file.size) throw new Error("This file is empty. Choose another image.");
    if (file.size > 12 * 1024 * 1024) throw new Error("The image is too large. Choose an image under 12 MB.");
    setUploadCheck("size", "passed", file.size < 1024 * 1024 ? `${Math.ceil(file.size / 1024)} KB` : `${(file.size / 1024 / 1024).toFixed(2)} MB`);
    currentCheck = "content";
    const bytes = new Uint8Array(await file.slice(0, 12).arrayBuffer());
    if (selectionId !== state.selectionId) return;
    const png = [137, 80, 78, 71, 13, 10, 26, 10].every((value, index) => bytes[index] === value);
    const jpeg = bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255;
    const mime = png ? "image/png" : jpeg ? "image/jpeg" : null;
    if (!mime || (extension === "png") !== png || (file.type && file.type !== "application/octet-stream" && file.type !== mime)) {
      throw new Error("The image contents do not match its file type. Export it as PNG or JPEG and try again.");
    }
    previewUrl = URL.createObjectURL(file);
    const decoded = new Image();
    decoded.src = previewUrl;
    try { await decoded.decode(); }
    catch (_) { throw new Error("This image could not be read. Choose an undamaged PNG, JPG or JPEG image."); }
    if (selectionId !== state.selectionId) return;
    setUploadCheck("content", "passed", "Image opened successfully");
    currentCheck = "dimensions";
    if (!decoded.naturalWidth || !decoded.naturalHeight) throw new Error("This image has invalid dimensions.");
    if (decoded.naturalWidth * decoded.naturalHeight > 16000000) {
      throw new Error("Image dimensions are too large. Choose an image with no more than 16 megapixels.");
    }
    if (selectionId !== state.selectionId) return;
    setUploadCheck("dimensions", "passed", `${decoded.naturalWidth} × ${decoded.naturalHeight} px`);
    state.file = file;
    showPreview(previewUrl);
    previewUrl = null;
    $("#file-name").textContent = file.name;
    const sizeLabel = file.size < 1024 * 1024 ? `${Math.ceil(file.size / 1024)} KB` : `${(file.size / 1024 / 1024).toFixed(2)} MB`;
    $("#file-meta").textContent = `${extension.toUpperCase()} · ${sizeLabel} · ${decoded.naturalWidth} × ${decoded.naturalHeight} px`;
    $("#selected-file").classList.remove("hidden");
    setNotice("Upload checks passed. Review the preview, then run verification.", "success");
  } catch (error) {
    if (selectionId === state.selectionId) {
      setUploadCheck(currentCheck, "error", "Check failed — see message below");
      setNotice(error.message);
    }
  } finally {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    if (selectionId === state.selectionId) { state.validating = false; updateUploadControls(); }
  }
}

function stageClass(stage) { return ["stage", stage.state].join(" "); }
function renderStages(job) {
  const visibleStages = (job.stages || []).filter((stage) => stage.key !== "reuse");
  $("#stage-list").innerHTML = visibleStages.map((stage, index) => {
    const icon = stage.state === "completed" ? "✓" : stage.state === "error" ? "!" : stage.state === "running" ? "•" : index + 1;
    return `<div class="${stageClass(stage)}"><div class="stage-top"><span class="stage-icon">${icon}</span><span>${stage.seconds ? `${stage.seconds}s` : ""}</span></div><span class="stage-name">${escapeHtml(stage.label)}</span><span class="stage-time">${escapeHtml(cleanStatus(stage.state))}</span></div>`;
  }).join("");
  const finished = visibleStages.filter((stage) => ["completed", "error", "skipped"].includes(stage.state)).length;
  const visibleProgress = ["completed", "completed_with_issues", "failed"].includes(job.status)
    ? 100 : Math.round((finished / Math.max(visibleStages.length, 1)) * 100);
  $("#progress-bar").style.width = `${visibleProgress}%`;
  $("#progress-value").textContent = `${visibleProgress}%`;
  const current = visibleStages.find((stage) => stage.state === "running");
  $("#progress-label").textContent = current ? current.label : cleanStatus(job.status);
}

function card(title, subtitle, status, body, wide = false, statusKind = "") {
  return `<article class="result-card${wide ? " wide" : ""}"><div class="result-card-head"><div><h3>${escapeHtml(title)}</h3><small>${escapeHtml(subtitle)}</small></div>${badge(status, statusKind)}</div><div class="result-body">${body}</div></article>`;
}

function renderResults(job) {
  const r = job.results || {};
  const cards = [];
  const ocr = r.ocr || {};
  cards.push(card("Headline extraction", "STEP 1 · OCR", ocr.status || "pending", ocr.extracted_text ? `<p class="quote">${escapeHtml(ocr.extracted_text)}</p>${rows([["Language", ocr.language || ocr.detected_language || "unknown"], ["Confidence", formatPercent(ocr.confidence)]])}` : `<p class="muted-text">${ocr.status ? "No headline text was returned. Try a sharper image with the main headline visible." : "Waiting for headline extraction…"}</p>`, true, ocr.error ? "bad" : ""));
  const claim = r.claim || {};
  cards.push(card("Headline / claim", "STEP 2 · CLAIM", claim.status || "pending", `<p class="quote">${escapeHtml(claim.original_claim || claim.headline || "No claim available")}</p>${rows([["English comparison", claim.english_claim || "Not available"], ["Claim source", claim.claim_source || claim.comparison_source || "OCR"]])}`));
  const visual = r.visual || {};
  cards.push(card("Visual image analysis", "STEP 3 · BLIP", visual.status || "pending", `<p class="muted-text">${escapeHtml(visual.description || visual.caption || visual.error || "Waiting for visual description")}</p>${rows([["Model", visual.model], ["Device", visual.device]])}`));
  const consistency = r.consistency || {};
  cards.push(card("Image–text consistency", "STEP 4 · CLIP", consistency.status || "pending", `<div class="metric-grid">${metric("Similarity", formatNumber(consistency.similarity_score, 4))}${metric("Alignment", consistency.alignment || consistency.label || "—")}${metric("Model", consistency.model || "—")}</div><p class="score-note">Similarity measures semantic association; it is not a probability that the news is true.</p>`));
  const authenticity = r.authenticity || {};
  const authenticityKind = String(authenticity.status || "").includes("high") || String(authenticity.status || "").includes("generated") ? "warn" : "";
  cards.push(card("Image authenticity", "STEP 5 · AI-generation indicator", authenticity.status || "pending", `<div class="metric-grid">${metric("Real image", formatPercent(authenticity.real_score))}${metric("AI-generated", formatPercent(authenticity.ai_generated_score ?? authenticity.ai_score))}${metric("Decision", authenticity.predicted_class || authenticity.prediction || authenticity.label || "—")}</div><p class="score-note">This is a model indicator for image-generation signals, not a definitive authenticity verdict.</p>`, false, authenticityKind));
  const context = r.context || {};
  cards.push(card("Context verification", "STEP 6 · EVENT / TIME / LOCATION", context.status || "pending", `<div class="metric-grid">${metric("Event", context.event_check || context.event_consistency || context.event || "—")}${metric("Time", context.time_check || context.time_consistency || context.time || "—")}${metric("Location", context.location_check || context.location_consistency || context.location || "—")}</div>${rows([["Evidence", context.evidence_status || context.evidence], ["Claim check", context.claim_check || context.claim_consistency || context.claim]])}`));
  const score = r.score || {};
  const hasScore = hasNumber(score.verification_score);
  const missing = Array.isArray(score.missing_factors) && score.missing_factors.length ? `No verified evidence for: ${score.missing_factors.join(", ")}.` : "Verified source evidence is required.";
  const scoreBody = hasScore ? `<div class="score-value">${formatNumber(score.verification_score, 1)} / 100</div>` : `<div class="score-pending"><strong>Pending</strong><span>${escapeHtml(missing)}</span></div>`;
  cards.push(card("Verification scoring", "STEP 7 · EVIDENCE-WEIGHTED", score.score_status || score.status || "pending", `${scoreBody}<p class="score-note">Coverage: ${formatPercent(score.context_evidence_coverage_percent ?? score.context_evidence_coverage)} · Risk: ${escapeHtml(score.risk_category || "undetermined")}</p>`, true, hasScore ? "" : "warn"));
  if (job.options && job.options.include_details) {
    const explanation = r.explanation || {};
    const evidence = r.evidence || {};
    cards.push(card("Explanation", "STEP 9 · XAI", explanation.status || "pending", `<p class="muted-text">${escapeHtml(explanation.summary || explanation.explanation || "No explanation available.")}</p>${Array.isArray(explanation.factors) && explanation.factors.length ? `<ul class="factor-list">${explanation.factors.map((factor) => `<li>${escapeHtml(factor)}</li>`).join("")}</ul>` : ""}`, false));
    cards.push(card("Evidence & references", "STEP 10 · RETRIEVAL", evidence.status || "pending", `<p class="muted-text">${escapeHtml(evidence.message || evidence.summary || "No verified evidence available.")}</p>${rows([["Records", evidence.record_count ?? evidence.records_found], ["Method", evidence.retrieval_method]])}`, false));
  }
  $("#results-grid").innerHTML = cards.join("") || `<div class="empty-results">Results will appear as the pipeline completes.</div>`;
}

function renderJob(job) {
  if (!job) return;
  $("#analysis-title").textContent = job.filename || "Verification run";
  $("#analysis-subtitle").textContent = job.status === "running" ? "The local models are processing the image…" : job.status === "completed" ? "Analysis complete. Review each signal below." : cleanStatus(job.status);
  renderStages(job); renderResults(job);
  if (["completed", "completed_with_issues", "failed"].includes(job.status)) {
    clearInterval(state.timer); state.timer = null;
    state.busy = false; updateUploadControls();
    $("#report-link").href = `/api/analyses/${encodeURIComponent(job.id)}/report`;
    $("#report-link").classList.remove("hidden");
  }
  if (job.error) setNotice(job.error);
}

async function pollJob(jobId) {
  try { const job = await request(`/api/analyses/${encodeURIComponent(jobId)}`); if (state.jobId === jobId) renderJob(job); }
  catch (error) { if (state.jobId !== jobId) return; clearInterval(state.timer); state.timer = null; setNotice(error.message); state.busy = false; updateUploadControls(); }
}

async function startAnalysis(event) {
  event.preventDefault(); setNotice("");
  if (state.busy || state.validating) return;
  if (!state.file) { setNotice("Choose an image first."); return; }
  const form = new FormData();
  form.append("file", state.file);
  state.busy = true; updateUploadControls();
  $("#report-link").classList.add("hidden");
  try {
    const job = await request("/api/analyses", { method: "POST", body: form });
    state.jobId = job.id;
    clearInterval(state.timer);
    $("#analysis-panel").classList.remove("hidden");
    renderJob(job);
    $("#analysis-panel").scrollIntoView({ behavior: "smooth", block: "start" });
    if (state.busy) state.timer = setInterval(() => pollJob(job.id), 1000);
  } catch (error) { state.busy = false; updateUploadControls(); setNotice(error.message); }
}

async function checkService() {
  try { await request("/api/health"); $("#service-status").textContent = "Local service ready"; $("#service-dot").classList.add("online"); }
  catch (_) { $("#service-status").textContent = "Start the local service"; $("#service-dot").classList.add("offline"); }
}

function resetAnalysis() { if (state.busy) return; clearInterval(state.timer); state.timer = null; state.jobId = null; resetSelection(); $("#analysis-panel").classList.add("hidden"); $("#results-grid").innerHTML = ""; setNotice(""); }

document.addEventListener("DOMContentLoaded", () => {
  checkService();
  const dropzone = $("#dropzone"); const input = $("#file-input");
  dropzone.addEventListener("click", () => { if (!state.busy) input.click(); }); dropzone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); if (!state.busy) input.click(); } });
  input.addEventListener("change", () => selectFile(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => {
    if (state.busy) return;
    if (event.dataTransfer.files.length !== 1) { resetAnalysis(); setNotice("Choose one PNG, JPG or JPEG image at a time."); return; }
    selectFile(event.dataTransfer.files[0]);
  });
  // Prevent a missed drop from navigating away from an active analysis.
  ["dragover", "drop"].forEach((name) => document.addEventListener(name, (event) => event.preventDefault()));
  $("#remove-file").addEventListener("click", () => { resetAnalysis(); dropzone.focus(); });
  $("#analysis-form").addEventListener("submit", startAnalysis); $("#reset-button").addEventListener("click", resetAnalysis);
  updateUploadControls();
});
