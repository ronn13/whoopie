/* ═══════════════════════════════════════════════════════════════════
   SafeAeroBERT Demo — app.js
   Handles API calls, rendering, and interactivity.
   ═══════════════════════════════════════════════════════════════════ */

'use strict';

// ── Example narratives ────────────────────────────────────────────────────────
const EXAMPLES = {
  eg1: `The aircraft departed runway 28L and climbed to cruising altitude. Approximately 45 minutes into the flight, the captain noticed unusual vibrations from the left engine. The autopilot disengaged unexpectedly due to an autothrottle malfunction. The first officer initiated an emergency descent while the flight crew contacted ATC for vectors to the nearest airport. Upon landing, inspection revealed the left engine had sustained severe compressor blade damage, resulting in a total write-off of the engine assembly. The aircraft sustained substantial damage and was declared a hull loss.`,

  eg2: `While on approach to runway 12, the controller issued a traffic advisory for a Cessna 172 crossing our path at 3,500 feet. The TCAS issued a resolution advisory commanding a climb. The captain initiated the evasive maneuver immediately. The two aircraft came within 400 feet vertical and 0.3 miles horizontal separation. No damage was sustained to either aircraft, and both crews filed safety reports following the incident.`,

  eg3: `During cruise flight at FL350, the right engine experienced a sudden loss of oil pressure followed by a compressor stall. The flight crew executed the engine failure checklist and shut down the engine. ATC provided priority handling and the aircraft landed safely on one engine. The engine was found to have a fractured turbine blade caused by metal fatigue. Minor damage was noted to the engine nacelle from debris.`,
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const narrativeInput   = document.getElementById('narrative-input');
const charCounter      = document.getElementById('char-counter');
const runBtn           = document.getElementById('run-btn');
const errorBanner      = document.getElementById('error-banner');
const errorMsg         = document.getElementById('error-msg');

const nerPlaceholder   = document.getElementById('ner-placeholder');
const nerResults       = document.getElementById('ner-results');
const nerLoading       = document.getElementById('ner-loading');
const highlightedText  = document.getElementById('highlighted-text');
const entityGroups     = document.getElementById('entity-groups');

// Model 1 (local SafeAeroBERT)
const classifyPH       = document.getElementById('classify-placeholder');
const classifyResults  = document.getElementById('classify-results');
const classifyLoading  = document.getElementById('classify-loading');
const verdictBox       = document.getElementById('verdict-box');
const verdictCategory  = document.getElementById('verdict-category');
const verdictLabel     = document.getElementById('verdict-label');
const verdictConf      = document.getElementById('verdict-conf');
const probBars         = document.getElementById('prob-bars');

// ── State ─────────────────────────────────────────────────────────────────────
let activeMode = 'pipeline';

// modelConfig[slot] = { slot, model_id, display_name, local, url }
const modelConfig = { 1: null, 2: null, 3: null };

// ── Init: load model config from server ───────────────────────────────────────
async function initConfig() {
  try {
    const cfg = await fetch('/api/config').then(r => r.json());
    for (const m of cfg.models) {
      modelConfig[m.slot] = m;
    }
  } catch (_) {
    // Server not running yet — slots stay null, UI will show placeholder
  }
  updateExternalPanels();
}

function updateExternalPanels() {
  [2, 3].forEach(slot => {
    const m = modelConfig[slot];
    const titleEl  = document.getElementById(`model-${slot}-title`);
    const tagEl    = document.getElementById(`model-${slot}-tag`);
    const phIcon   = document.getElementById(`model-${slot}-ph-icon`);
    const phText   = document.getElementById(`model-${slot}-ph-text`);

    if (!m || !m.url) {
      // Not yet configured
      titleEl.textContent = m ? m.display_name : `Model ${slot}`;
      tagEl.textContent   = 'not configured';
      phIcon.textContent  = '🕒';
      phText.textContent  = 'Awaiting teammate integration';
    } else {
      titleEl.textContent = m.display_name;
      tagEl.textContent   = 'external · ready';
      phIcon.textContent  = '📊';
      phText.textContent  = 'Run the pipeline to see results.';
    }
  });
}

// ── Char counter ──────────────────────────────────────────────────────────────
narrativeInput.addEventListener('input', () => {
  const n = narrativeInput.value.length;
  charCounter.textContent = `${n.toLocaleString()} char${n !== 1 ? 's' : ''}`;
});

// ── Example chips ─────────────────────────────────────────────────────────────
['eg1', 'eg2', 'eg3'].forEach(key => {
  document.getElementById(`${key}-btn`).addEventListener('click', () => {
    narrativeInput.value = EXAMPLES[key];
    narrativeInput.dispatchEvent(new Event('input'));
    narrativeInput.focus();
  });
});

// ── Mode tabs ─────────────────────────────────────────────────────────────────
document.querySelectorAll('.mode-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    activeMode = tab.dataset.mode;
  });
});

// ── Run button ────────────────────────────────────────────────────────────────
runBtn.addEventListener('click', runAnalysis);
narrativeInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) runAnalysis();
});

async function runAnalysis() {
  const text = narrativeInput.value.trim();
  if (!text) { showError('Please paste a narrative before running.'); return; }
  hideError();
  setRunning(true);

  try {
    if (activeMode === 'pipeline') {
      showNerLoading(true);
      showClassifyLoading(true, 1);
      showClassifyLoading(true, 2);
      showClassifyLoading(true, 3);

      // Run NER + multi-model classification concurrently
      const [pipelineData, multiData] = await Promise.all([
        post('/api/pipeline', { text }),
        post('/api/multi-predict', { narrative: text }),
      ]);

      renderNer(text, pipelineData.ner);
      renderModelResult(1, multiData['1']);
      renderModelResult(2, multiData['2']);
      renderModelResult(3, multiData['3']);

    } else if (activeMode === 'ner') {
      showNerLoading(true);
      resetClassify(1); resetClassify(2); resetClassify(3);
      const data = await post('/api/extract', { text });
      renderNer(text, data);

    } else {
      // classify-only mode
      showClassifyLoading(true, 1);
      showClassifyLoading(true, 2);
      showClassifyLoading(true, 3);
      resetNer();
      const multiData = await post('/api/multi-predict', { narrative: text });
      renderModelResult(1, multiData['1']);
      renderModelResult(2, multiData['2']);
      renderModelResult(3, multiData['3']);
    }
  } catch (err) {
    showError(err.message || 'Server error. Make sure the Flask server is running on port 5050.');
    resetNer();
    resetClassify(1); resetClassify(2); resetClassify(3);
  } finally {
    setRunning(false);
    showNerLoading(false);
    showClassifyLoading(false, 1);
    showClassifyLoading(false, 2);
    showClassifyLoading(false, 3);
  }
}

// ── Dispatch render per slot ──────────────────────────────────────────────────
function renderModelResult(slot, data) {
  if (slot === 1) {
    if (data && data.error) {
      // Fall back to showing error in model-1 panel
      resetClassify(1);
      return;
    }
    renderClassify(data, 1);
    return;
  }

  // Slots 2 & 3
  const errEl  = document.getElementById(`classify-error-${slot}`);
  const errMsg = document.getElementById(`error-msg-${slot}`);
  const ph     = document.getElementById(`classify-placeholder-${slot}`);
  const res    = document.getElementById(`classify-results-${slot}`);

  if (!data || data.error) {
    const msg = (!data || data.error === 'not_configured')
      ? 'Not configured — set MODEL_' + slot + '_URL env var'
      : (data.error || 'Unknown error');
    ph.classList.add('hidden');
    res.classList.add('hidden');
    errMsg.textContent = msg;
    errEl.classList.remove('hidden');
    return;
  }

  errEl.classList.add('hidden');
  renderClassify(data, slot);
}

// ── API helpers ───────────────────────────────────────────────────────────────
async function post(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const json = await res.json();
  if (!res.ok || json.error) throw new Error(json.error || `HTTP ${res.status}`);
  return json;
}

// ── NER rendering ─────────────────────────────────────────────────────────────
function renderNer(text, data) {
  const spans   = data.spans || [];
  const actors  = data.ACTOR   || [];
  const systems = data.SYSTEM  || [];
  const triggers= data.TRIGGER || [];

  const sorted = [...spans].sort((a, b) => a.start - b.start || (b.end - b.start) - (a.end - a.start));
  const noOverlap = [];
  let cursor = 0;
  for (const sp of sorted) {
    if (sp.start >= cursor) {
      noOverlap.push(sp);
      cursor = sp.end;
    }
  }

  let html = '';
  let pos = 0;
  for (const sp of noOverlap) {
    if (sp.start > pos) html += escHtml(text.slice(pos, sp.start));
    html += `<mark class="ent" data-type="${sp.type}">${escHtml(text.slice(sp.start, sp.end))}</mark>`;
    pos = sp.end;
  }
  if (pos < text.length) html += escHtml(text.slice(pos));

  highlightedText.innerHTML = html;

  entityGroups.innerHTML = buildEntityGroup('actor',   'ACTOR',   actors)
                         + buildEntityGroup('system',  'SYSTEM',  systems)
                         + buildEntityGroup('trigger', 'TRIGGER', triggers);

  nerPlaceholder.classList.add('hidden');
  nerResults.classList.remove('hidden');
  nerResults.classList.add('fade-in');
}

function buildEntityGroup(cls, label, items) {
  const chips = items.length
    ? items.map(t => `<span class="entity-chip ${cls}">${escHtml(t)}</span>`).join('')
    : `<span class="entity-chip empty">none detected</span>`;
  return `
    <div class="entity-group">
      <div class="entity-group-label ${cls}">${label}</div>
      <div class="entity-chips">${chips}</div>
    </div>`;
}

function resetNer() {
  nerResults.classList.add('hidden');
  nerPlaceholder.classList.remove('hidden');
  nerPlaceholder.classList.remove('fade-in');
}

// ── Classifier rendering ──────────────────────────────────────────────────────
const CATEGORY_COLOR = {
  'MAC':   '#f87171',
  'CFIT':  '#f97316',
  'GCOL':  '#facc15',
  'SEC':   '#a78bfa',
  'LOC-I': '#ef4444',
  'ATM':   '#38bdf8',
  'TURB':  '#4ade80',
  'USOS':  '#fb923c',
  'RE':    '#f43f5e',
  'OTHR':  '#94a3b8',
  'UNK':   '#94a3b8',
};

/**
 * Render classification result into the panel for the given slot (1, 2, or 3).
 * `data` must follow the API contract:
 *   { model_id, display_name, prediction: { top_class, confidence, top_5 }, inference_time_ms }
 * or the flat form returned by /api/classify (no outer wrapper).
 */
function renderClassify(data, slot) {
  slot = slot || 1;

  // Resolve IDs
  const isSlot1 = slot === 1;
  const vBox    = isSlot1 ? verdictBox      : document.getElementById(`verdict-box-${slot}`);
  const vCat    = isSlot1 ? verdictCategory : document.getElementById(`verdict-category-${slot}`);
  const vLbl    = isSlot1 ? verdictLabel    : document.getElementById(`verdict-label-${slot}`);
  const vConf   = isSlot1 ? verdictConf     : document.getElementById(`verdict-conf-${slot}`);
  const pBars   = isSlot1 ? probBars        : document.getElementById(`prob-bars-${slot}`);
  const ph      = isSlot1 ? classifyPH      : document.getElementById(`classify-placeholder-${slot}`);
  const res     = isSlot1 ? classifyResults : document.getElementById(`classify-results-${slot}`);

  // Unwrap prediction envelope if present
  const predData = data.prediction ? data.prediction : data;
  const topClassFull = predData.top_class || 'Unknown';
  const topClassAbbr = (predData.top_5 && predData.top_5.length > 0)
    ? (predData.top_5[0].class || topClassFull)
    : topClassFull;

  // Update model title for slots 2/3 if display_name is provided
  if (!isSlot1 && data.display_name) {
    const titleEl = document.getElementById(`model-${slot}-title`);
    if (titleEl) titleEl.textContent = data.display_name;
  }

  vBox.dataset.cat = topClassAbbr;
  vCat.textContent = topClassAbbr;
  vCat.style.color = CATEGORY_COLOR[topClassAbbr] || '#e2e8f0';
  vLbl.textContent = topClassFull;
  vConf.textContent = `Confidence: ${(predData.confidence * 100).toFixed(1)}%`;

  const top5 = predData.top_5 || [];
  pBars.innerHTML = top5.map((p, i) => {
    const isTop = i === 0;
    const pct   = (p.confidence * 100).toFixed(1);
    return `
      <div class="prob-row">
        <span class="prob-label">${escHtml(p.class)}</span>
        <div class="prob-bar-wrap" title="${escHtml(p.class)}">
          <div class="prob-bar-fill${isTop ? ' top' : ''}" style="width:0" data-width="${pct}%"></div>
        </div>
        <span class="prob-val">${pct}%</span>
      </div>`;
  }).join('');

  // Inference time
  if (!isSlot1) {
    const timeEl = document.getElementById(`inference-time-${slot}`);
    if (timeEl && data.inference_time_ms != null) {
      timeEl.textContent = `Inference: ${data.inference_time_ms} ms`;
    }
  }

  ph.classList.add('hidden');
  res.classList.remove('hidden');
  res.classList.add('fade-in');

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      pBars.querySelectorAll('.prob-bar-fill').forEach(el => {
        el.style.width = el.dataset.width;
      });
    });
  });
}

function resetClassify(slot) {
  slot = slot || 1;
  const isSlot1 = slot === 1;
  const res  = isSlot1 ? classifyResults : document.getElementById(`classify-results-${slot}`);
  const ph   = isSlot1 ? classifyPH      : document.getElementById(`classify-placeholder-${slot}`);
  const errEl = !isSlot1 ? document.getElementById(`classify-error-${slot}`) : null;

  res.classList.add('hidden');
  ph.classList.remove('hidden');
  if (errEl) errEl.classList.add('hidden');
}

// ── Loading helpers ───────────────────────────────────────────────────────────
function showNerLoading(on) {
  nerLoading.classList.toggle('hidden', !on);
}

function showClassifyLoading(on, slot) {
  slot = slot || 1;
  const isSlot1 = slot === 1;
  if (isSlot1) {
    classifyLoading.classList.toggle('hidden', !on);
  } else {
    document.getElementById(`classify-loading-${slot}`).classList.toggle('hidden', !on);
  }
}

function setRunning(on) {
  runBtn.disabled = on;
  runBtn.querySelector('.run-btn-text').textContent = on ? 'Analyzing…' : 'Analyze Narrative';
}

// ── Error helpers ─────────────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.textContent = msg;
  errorBanner.classList.remove('hidden');
}
function hideError() {
  errorBanner.classList.add('hidden');
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
initConfig();
