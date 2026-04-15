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
      showClassifyLoading(true);
      const data = await post('/api/pipeline', { text });
      renderNer(text, data.ner);
      renderClassify(data.classification);

    } else if (activeMode === 'ner') {
      showNerLoading(true);
      resetClassify();
      const data = await post('/api/extract', { text });
      renderNer(text, data);

    } else {
      showClassifyLoading(true);
      resetNer();
      const data = await post('/api/classify', { text });
      renderClassify(data);
    }
  } catch (err) {
    showError(err.message || 'Server error. Make sure the Flask server is running on port 5050.');
    resetNer();
    resetClassify();
  } finally {
    setRunning(false);
    showNerLoading(false);
    showClassifyLoading(false);
  }
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

  // Build highlighted HTML from spans
  // Sort spans by start position, handle overlaps by taking longest non-overlapping
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

  // Entity chip groups
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
  'MAC':   '#f87171', // Red
  'CFIT':  '#f97316', // Orange
  'GCOL':  '#facc15', // Yellow
  'SEC':   '#a78bfa', // Purple
  'LOC-I': '#ef4444', // Red-ish
  'ATM':   '#38bdf8', // Light Blue
  'TURB':  '#4ade80', // Green
  'USOS':  '#fb923c', // Orange-ish
  'RE':    '#f43f5e', // Rose
  'OTHR':  '#94a3b8', // Gray
  'UNK':   '#94a3b8'  // Gray
};

function renderClassify(data) {
  // Try to grab top_class and top_5 from data directly or from data.prediction if nested
  const predData = data.prediction ? data.prediction : data;
  const topClassFull = predData.top_class || 'Unknown';
  
  // Create a short abbreviation for visual display if possible (e.g. "CFIT" from the top_5 object)
  let topClassAbbr = "OTHR";
  if (predData.top_5 && predData.top_5.length > 0) {
     // We try to reverse lookup the class if needed, or simply extract the abbreviation if provided
     // We'll just display the full text as the abbreviation if it's not present, or use the first element's abbr
     topClassAbbr = predData.top_5[0].class || topClassFull;
  }

  verdictBox.dataset.cat = topClassAbbr;
  verdictCategory.textContent = topClassAbbr;
  verdictCategory.style.color = CATEGORY_COLOR[topClassAbbr] || '#e2e8f0';
  verdictLabel.textContent  = topClassFull;
  verdictConf.textContent   = `Confidence: ${(predData.confidence * 100).toFixed(1)}%`;

  // Probability bars
  const top_5 = predData.top_5 || [];
  probBars.innerHTML = top_5.map((p, i) => {
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

  classifyPH.classList.add('hidden');
  classifyResults.classList.remove('hidden');
  classifyResults.classList.add('fade-in');

  // Animate bars after render
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      probBars.querySelectorAll('.prob-bar-fill').forEach(el => {
        el.style.width = el.dataset.width;
      });
    });
  });
}

function resetClassify() {
  classifyResults.classList.add('hidden');
  classifyPH.classList.remove('hidden');
}

// ── Loading helpers ───────────────────────────────────────────────────────────
function showNerLoading(on) {
  nerLoading.classList.toggle('hidden', !on);
}
function showClassifyLoading(on) {
  classifyLoading.classList.toggle('hidden', !on);
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
