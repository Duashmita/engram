// render.js — DOM updates from state.
//
// Renderer is idempotent: given the same state it produces the same DOM.
// This is what makes scrubbing work — the replay loop calls rebuild() and
// then renderAll() and we don't have to worry about cleaning up animations
// that fired in the alternate timeline.
//
// Animation hints (state.flash_ff, fresh_mem_id, etc.) are *consumed* —
// the renderer reads them, applies a transient class, and clears them.

const TRAITS = ['O','C','E','A','N'];

// ---------- radar (Chart.js) ----------
let radarChart = null;

export function initRadar(canvas) {
  radarChart = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: TRAITS,
      datasets: [
        {
          label: 'baseline',
          data: [0,0,0,0,0],
          borderColor: 'rgba(122,134,153,0.7)',
          backgroundColor: 'rgba(122,134,153,0.05)',
          borderDash: [4,3],
          borderWidth: 1.5,
          pointRadius: 0,
        },
        {
          label: 'effective',
          data: [0,0,0,0,0],
          borderColor: 'rgba(106,163,255,1)',
          backgroundColor: 'rgba(106,163,255,0.18)',
          borderWidth: 2,
          pointRadius: 3,
          pointBackgroundColor: 'rgba(106,163,255,1)',
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 280 },
      plugins: { legend: { display: false } },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { display: false, stepSize: 0.2 },
          grid:  { color: 'rgba(45,55,68,0.7)' },
          angleLines: { color: 'rgba(45,55,68,0.7)' },
          pointLabels: { color: '#d6deeb', font: { size: 11 } },
        },
      },
    },
  });
}

export function renderRadar(state) {
  if (!radarChart) return;
  radarChart.data.datasets[0].data = TRAITS.map(t => state.baseline_ocean[t]);
  radarChart.data.datasets[1].data = TRAITS.map(t => state.effective_ocean[t]);
  radarChart.update('none');

  // ocean readout numerals
  const out = document.getElementById('ocean-readout');
  out.innerHTML = TRAITS.map(t =>
    `${t} <span class="v">${state.effective_ocean[t].toFixed(2)}</span>`
  ).join('  ');

  // fight/flight pulse (consume hint)
  if (state.flash_ff) {
    const wrap = document.getElementById('radar-wrap');
    wrap.classList.remove('flash');
    void wrap.offsetWidth; // force reflow so the animation re-triggers
    wrap.classList.add('flash');
    state.flash_ff = false;
  }
}

// ---------- header strip ----------
export function renderHeader(state) {
  document.getElementById('npc-name').textContent = state.npc_name;
  document.getElementById('persona').textContent  = state.persona ?? '';
}

// ---------- pipeline turns (Panel B) ----------
export function renderTurns(state, opts = {}) {
  const list = document.getElementById('turn-list');
  // newest-first reads better — newest events animate at the top, the user's
  // eye lands there. Document choice: newest-on-top.
  const turns = [...state.turns].reverse();

  // Diff strategy: rebuild HTML if turn count changed; otherwise patch in place.
  const have = list.children.length;
  if (have !== turns.length) {
    list.innerHTML = turns.map(t => turnCardHTML(t, state)).join('');
    bindTurnHandlers(list);
  } else {
    turns.forEach((t, idx) => {
      const card = list.children[idx];
      if (!card) return;
      const sig = turnSignature(t);
      if (card.dataset.sig !== sig) {
        card.outerHTML = turnCardHTML(t, state);
      }
    });
    bindTurnHandlers(list);
  }

  // mark the active (most-recent) turn
  Array.from(list.children).forEach((c, idx) => {
    c.classList.toggle('active', idx === 0 && !state.session_ended);
  });
}

function turnSignature(t) {
  // any change in stage state changes the signature
  return [
    t.threat ? `T${t.threat.is_threat?'1':'0'}${t.threat.magnitude.toFixed(2)}` : 'T-',
    t.retrieval ? `R${t.retrieval.scored.length}/${t.retrieval.selected_ids.length}` : 'R-',
    t.mode ?? 'M-',
    t.contradiction ? `C${t.contradiction.stages.length}/${t.contradiction.conflicts.length}` : 'C-',
    t.response ? `S${t.response.attempt}` : 'S-',
    t.consolidated ? 'X1' : 'X-',
  ].join('|');
}

function turnCardHTML(t, state) {
  const sig = turnSignature(t);

  // cell: threat
  const threatCell = !t.threat
    ? `<div class="cell empty">threat</div>`
    : `<div class="cell fired"><span class="pill ${t.threat.is_threat?'pill-bad':'pill-good'}">
         ${t.threat.is_threat ? 'threat' : 'safe'} ${t.threat.magnitude.toFixed(2)}
       </span></div>`;

  // cell: retrieval
  const retCell = !t.retrieval
    ? `<div class="cell empty">retrieve</div>`
    : (() => {
        const total = t.retrieval.scored.length;
        const qual  = t.retrieval.scored.filter(s => s.qualified).length;
        return `<div class="cell fired"><span class="pill pill-info">${qual}/${total} · ${t.retrieval.mode}</span></div>`;
      })();

  // cell: mode
  const modeCell = !t.mode
    ? `<div class="cell empty">mode</div>`
    : `<div class="cell fired"><span class="pill pill-mode-${t.mode}">${t.mode.replace('_','/')}</span></div>`;

  // cell: contradiction
  let conCell;
  if (!t.contradiction) {
    conCell = `<div class="cell empty">check</div>`;
  } else {
    const has = t.contradiction.conflicts.length > 0;
    const stagesText = t.contradiction.stages.join('/');
    conCell = `<div class="cell fired"><span class="pill ${has?'pill-warn':'pill-good'}">
         ${has ? '⚠ '+t.contradiction.conflicts.length : '✓'} ${stagesText}
       </span></div>`;
  }

  // cell: response
  const respCell = !t.response
    ? `<div class="cell empty">respond</div>`
    : `<div class="cell fired">
         <span class="snippet" title="${escapeHTML(t.response.text)}">${escapeHTML(t.response.text)}</span>
         ${t.response.attempt > 1 ? `<span class="pill pill-warn" style="margin-left:4px">re-roll</span>` : ''}
       </div>`;

  // cell: consolidated
  const consCell = !t.consolidated
    ? `<div class="cell empty">save</div>`
    : `<div class="cell fired"><span class="pill pill-plus">+1 mem</span></div>`;

  // expansion (retrieval table)
  const expand = t.retrieval ? retrievalTableHTML(t.retrieval) : '';

  return `
    <div class="turn-card" data-turn="${t.turn}" data-sig="${sig}">
      <div class="row">
        <div class="turn-num">#${t.turn}</div>
        ${threatCell}${retCell}${modeCell}${conCell}${respCell}${consCell}
      </div>
      <div class="turn-expand">${expand}</div>
    </div>`;
}

function retrievalTableHTML(r) {
  const sorted = [...r.scored].sort((a,b) => b.score - a.score);
  if (sorted.length === 0) {
    return `<div class="muted">No memories scored (instinct mode entered).</div>`;
  }
  const maxScore = Math.max(r.threshold * 1.4, ...sorted.map(s => s.score));
  const thresholdPct = ((r.threshold / maxScore) * 100).toFixed(1);

  const rows = sorted.map(s => {
    const sel = r.selected_ids.includes(s.id);
    const fillPct = ((s.score / maxScore) * 100).toFixed(1);
    return `
      <div class="row ${sel?'selected':''}">
        <div class="text-cell" title="${escapeHTML(s.text ?? '')}">${escapeHTML(truncate(s.text ?? '', 80))}</div>
        <div>rag ${(s.rag ?? 0).toFixed(2)}</div>
        <div>OΣ ${(s.ocean_sum ?? 0).toFixed(2)}</div>
        <div>imp ${s.importance ?? 0}</div>
        <div>sc ${(s.score ?? 0).toFixed(1)}</div>
        <div class="check">${sel?'●':''}</div>
        <div class="bar-fill ${s.qualified?'qualified':''}" style="width:${fillPct}%"></div>
      </div>`;
  }).join('');

  return `
    <div class="retrieval-table" style="--threshold-pct:${thresholdPct}%;">
      <div class="h">memory</div><div class="h">rag</div><div class="h">OΣ</div><div class="h">imp</div><div class="h">score</div><div class="h"></div>
      ${rows}
      <div class="score-bar"></div>
    </div>`;
}

function bindTurnHandlers(list) {
  Array.from(list.children).forEach(card => {
    if (card._bound) return;
    card._bound = true;
    card.querySelector('.row').addEventListener('click', () => {
      card.classList.toggle('expanded');
    });
  });
}

// ---------- memory store (Panel C) ----------
export function renderMemoryStore(state) {
  // Session window — render exactly window_size slots
  const track = document.getElementById('window-track');
  const ws = state.window_size || 7;
  const slots = [];
  const visible = state.session_window.slice(-ws);
  for (let i = 0; i < ws; i++) {
    const turn = visible[i];
    if (!turn) {
      slots.push(`<div class="slot empty"></div>`);
    } else {
      const fresh = (i === visible.length - 1) ? 'fresh' : '';
      slots.push(`<div class="slot ${fresh}">
        <div class="p">› ${escapeHTML(truncate(turn.player, 40))}</div>
        <div class="n">‹ ${escapeHTML(truncate(turn.npc, 40))}</div>
      </div>`);
    }
  }
  track.innerHTML = slots.join('');
  document.getElementById('window-count').textContent = `${visible.length}/${ws}`;

  // summaries
  const sumEl = document.getElementById('summaries');
  if (state.summaries.length) {
    sumEl.innerHTML = state.summaries.slice(-3).map(s =>
      `<div class="summary-item">${escapeHTML(s)}</div>`
    ).join('');
  } else {
    sumEl.innerHTML = '';
  }

  // all-memories list
  const memEl = document.getElementById('mem-list');
  document.getElementById('mem-count').textContent = state.memories.length;

  // diff: only re-render if count changes or selected_ids changed.
  // We track via a data attribute.
  const sig = `${state.memories.length}|${state.selected_ids.join(',')}`;
  if (memEl.dataset.sig !== sig) {
    // newest first
    const items = [...state.memories].reverse().map(m => {
      const isFresh = (m.id === state.fresh_mem_id) ? 'fresh' : '';
      const isRet = state.selected_ids.includes(m.id) ? 'retrieved' : '';
      return `<div class="mem-item ${isFresh} ${isRet}" data-id="${m.id}">
        <div class="meta">
          <span class="src">${m.source}</span>
          <span class="imp">imp ${m.importance}</span>
          <span>${shortId(m.id)}</span>
        </div>
        <div class="text" title="${escapeHTML(m.text)}">${escapeHTML(truncate(m.text, 80))}</div>
      </div>`;
    }).join('');
    memEl.innerHTML = items;
    memEl.dataset.sig = sig;
    state.fresh_mem_id = null;
  }

  // key memories
  const keyEl = document.getElementById('key-list');
  if (state.key_memories?.ids?.length) {
    const fresh = state.fresh_key ? 'fresh' : '';
    keyEl.innerHTML = state.key_memories.ids.map(id => {
      const m = state.memories.find(x => x.id === id);
      const text = m?.text ?? id;
      return `<div class="key-item ${fresh}">${escapeHTML(truncate(text, 80))}</div>`;
    }).join('');
    state.fresh_key = false;
  } else {
    keyEl.innerHTML = '';
  }

  // facts
  const factsEl = document.getElementById('facts-list');
  const recent = state.facts.slice(-30);
  factsEl.innerHTML = recent.map((f, idx) => {
    const fresh = (idx === recent.length - 1 && state.fresh_fact_idx != null) ? 'fresh' : '';
    if (f.kind === 'asserted') return `<div class="fact-item asserted ${fresh}">${escapeHTML(f.fact)}</div>`;
    if (f.kind === 'revised')  return `<div class="fact-item revised">${f.old?`<span class="strike">${escapeHTML(f.old)}</span>`:''}${escapeHTML(f.fact)}</div>`;
    if (f.kind === 'rejected') return `<div class="fact-item rejected"><span class="strike">${escapeHTML(f.fact)}</span></div>`;
    return '';
  }).join('');
  state.fresh_fact_idx = null;
}

// ---------- transcript (Panel D) ----------
export function renderTranscript(state) {
  const el = document.getElementById('transcript');
  const lastTurn = state.current_turn;
  el.innerHTML = state.transcript.map(line => {
    const cur = (line.turn === lastTurn) ? 'current' : '';
    const tag = line.who === 'npc' && line.mode
      ? `<span class="mode-tag pill-mode-${line.mode}">${line.mode}</span>` : '';
    return `<div class="line ${line.who} ${cur}">
      <span class="who">${line.who}</span>
      <span class="what">${escapeHTML(line.text)}${tag}</span>
    </div>`;
  }).join('');
  // scroll to bottom (current playback)
  el.scrollTop = el.scrollHeight;
}

// ---------- top-level orchestrator ----------
export function renderAll(state) {
  renderHeader(state);
  renderRadar(state);
  renderTurns(state);
  renderMemoryStore(state);
  renderTranscript(state);
}

// ---------- helpers ----------
function escapeHTML(s) {
  return String(s ?? '')
    .replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;')
    .replaceAll('"','&quot;').replaceAll("'",'&#39;');
}
function truncate(s, n) {
  s = String(s ?? '');
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}
function shortId(id) {
  if (!id) return '';
  return id.length > 10 ? id.slice(0, 8) + '…' : id;
}
