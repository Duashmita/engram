// live.js — Live Chat mode for Engram viz.
//
// Architecture:
//   * Replay mode reads events from a static NDJSON file and emits them on a
//     virtual clock. Live mode is the same shape, but events stream from the
//     backend over Server-Sent Events and we drive the same `apply()` reducer
//     and `renderAll()` renderer used by replay.
//   * The backend exposes /start, /turn, /end, /health (see API contract in
//     the build spec). All bodies/responses are JSON; /turn responds with
//     `text/event-stream` framed events that match the existing event schema.
//   * This module owns the live-mode UI panel (NPC picker + Start), the
//     composer (textarea + Send), and the SSE-parsing turn loop. State and
//     renderer are passed in from app.js so we don't fork ownership.

import { BACKEND_URL } from '../config.js';
import { freshState } from './state.js';

// NPC presets shown in the picker. Names come from the existing personas.
const NPC_PRESETS = [
  { id: 'jeanie',   name: 'Jeanie' },
  { id: 'guard',    name: 'Paranoid Guard' },
  { id: 'merchant', name: 'Friendly Merchant' },
  { id: 'clerk',    name: 'Rigid Clerk' },
  { id: 'maya',     name: 'Maya' },
  { id: 'hale',     name: 'Hale' },
];

// Module-local state ----------------------------------------------------------
let stateRef = null;          // { state }   — shared mutable handle from app.js
let applyFn  = null;          // (state, ev) => state
let renderFn = null;          // (state)     => void

let currentSessionId = null;
let npcId            = null;
let inFlight         = false; // one /turn at a time
let beaconInstalled  = false;

// LocalStorage keys
const LS_SESSION = 'engram_live_session_id';
const LS_NPC     = 'engram_live_npc_id';
const LS_KEY     = 'engram_gemini_key';

// ---------------- public entrypoint -----------------------------------------

/**
 * Initialize live-mode UI. Idempotent: safe to call repeatedly when toggling
 * from replay back into live.
 */
export function enterLive({ stateRef: sRef, applyFn: aFn, renderFn: rFn }) {
  stateRef = sRef;
  applyFn  = aFn;
  renderFn = rFn;

  showLiveUI();
  installBeacon();
  wireComposer();
  wireStartButton();
  wireSettingsDialog();

  // Try to resume a previous session if one was persisted.
  const savedSession = localStorage.getItem(LS_SESSION);
  const savedNpc     = localStorage.getItem(LS_NPC);
  if (savedSession && savedNpc) {
    currentSessionId = savedSession;
    npcId            = savedNpc;
    setSelectedNpc(savedNpc);
    setStatus(`resuming session ${shortId(savedSession)}…`);
    setComposerEnabled(true);
  } else {
    setStatus('pick an NPC and start a session');
    setComposerEnabled(false);
  }
}

/** Tear-down hook for when user toggles back to replay. Clears live UI. */
export function exitLive() {
  hideLiveUI();
  // We deliberately do NOT /end the session — user may toggle back.
}

// ---------------- UI plumbing -----------------------------------------------

function showLiveUI() {
  // Replay session picker hides while live is active.
  const replayPicker = document.querySelector('.session-picker');
  if (replayPicker) replayPicker.classList.add('hidden');

  // Live picker + Start
  const livePicker = document.getElementById('live-picker');
  if (livePicker) {
    livePicker.classList.remove('hidden');
    // populate options once
    const sel = document.getElementById('live-npc-select');
    if (sel && sel.options.length === 0) {
      for (const p of NPC_PRESETS) {
        const opt = document.createElement('option');
        opt.value = p.id;
        opt.textContent = p.name;
        sel.appendChild(opt);
      }
    }
  }

  // Composer (slide-up via class)
  const composer = document.getElementById('composer');
  if (composer) {
    composer.classList.remove('hidden');
    requestAnimationFrame(() => composer.classList.add('visible'));
  }

  // Hide replay-only controls (transport buttons + scrubber).
  document.querySelector('.controls')?.classList.add('hidden');
  document.querySelector('.scrubber-row')?.classList.add('hidden');
}

function hideLiveUI() {
  document.querySelector('.session-picker')?.classList.remove('hidden');
  document.getElementById('live-picker')?.classList.add('hidden');
  const composer = document.getElementById('composer');
  if (composer) {
    composer.classList.remove('visible');
    composer.classList.add('hidden');
  }
  document.querySelector('.controls')?.classList.remove('hidden');
  document.querySelector('.scrubber-row')?.classList.remove('hidden');
}

function wireStartButton() {
  const btn = document.getElementById('live-start');
  if (!btn || btn._wired) return;
  btn._wired = true;
  btn.addEventListener('click', startSession);
}

function wireComposer() {
  const send  = document.getElementById('composer-send');
  const input = document.getElementById('composer-input');
  if (send && !send._wired)   { send._wired = true;  send.addEventListener('click', onSendClick); }
  if (input && !input._wired) {
    input._wired = true;
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        onSendClick();
      }
    });
    // auto-grow up to 4 rows
    input.addEventListener('input', () => {
      input.style.height = 'auto';
      const max = parseFloat(getComputedStyle(input).lineHeight) * 4 + 12;
      input.style.height = Math.min(input.scrollHeight, max) + 'px';
    });
  }
}

function wireSettingsDialog() {
  const open  = document.getElementById('btn-settings');
  const dlg   = document.getElementById('settings-dialog');
  const save  = document.getElementById('settings-save');
  const clear = document.getElementById('settings-clear');
  const close = document.getElementById('settings-close');
  const input = document.getElementById('settings-key-input');
  if (!dlg) return;

  if (open && !open._wired) {
    open._wired = true;
    open.addEventListener('click', () => {
      input.value = localStorage.getItem(LS_KEY) ?? '';
      dlg.showModal();
    });
  }
  if (save && !save._wired) {
    save._wired = true;
    save.addEventListener('click', e => {
      e.preventDefault();
      const v = (input.value ?? '').trim();
      if (v) localStorage.setItem(LS_KEY, v);
      else   localStorage.removeItem(LS_KEY);
      dlg.close();
    });
  }
  if (clear && !clear._wired) {
    clear._wired = true;
    clear.addEventListener('click', e => {
      e.preventDefault();
      localStorage.removeItem(LS_KEY);
      input.value = '';
    });
  }
  if (close && !close._wired) {
    close._wired = true;
    close.addEventListener('click', e => { e.preventDefault(); dlg.close(); });
  }
}

function installBeacon() {
  if (beaconInstalled) return;
  beaconInstalled = true;
  window.addEventListener('beforeunload', () => {
    if (!currentSessionId) return;
    try {
      const blob = new Blob([JSON.stringify({ session_id: currentSessionId })],
                            { type: 'application/json' });
      navigator.sendBeacon(`${BACKEND_URL}/end`, blob);
    } catch (_) { /* best-effort */ }
  });
}

function setSelectedNpc(id) {
  const sel = document.getElementById('live-npc-select');
  if (sel) sel.value = id;
}

// ---------------- network: /start -------------------------------------------

async function startSession() {
  const sel = document.getElementById('live-npc-select');
  const chosen = sel?.value ?? NPC_PRESETS[0].id;

  setStatus('starting session…');
  setComposerEnabled(false);

  const headers = { 'Content-Type': 'application/json' };
  const key = localStorage.getItem(LS_KEY);
  if (key) headers['X-Gemini-Key'] = key;
  const body = JSON.stringify({ npc_id: chosen, ...(key ? { gemini_key: key } : {}) });

  let res;
  try {
    res = await fetch(`${BACKEND_URL}/start`, { method: 'POST', headers, body });
  } catch (err) {
    setStatus(`network error starting: ${err.message}`);
    return;
  }

  if (res.status === 429) {
    const j = await safeJSON(res);
    setStatus(`rate limited — try again in ${j?.retry_after_s ?? '?'}s`);
    return;
  }
  if (res.status === 503) { setStatus('server has no API key configured (set yours in Settings)'); return; }
  if (res.status === 400) { setStatus('unknown NPC preset'); return; }
  if (!res.ok)            { setStatus(`start failed: ${res.status}`); return; }

  const data = await res.json();
  currentSessionId = data.session_id;
  npcId            = chosen;
  localStorage.setItem(LS_SESSION, currentSessionId);
  localStorage.setItem(LS_NPC,     npcId);

  // Reset state and inject session_init from the header payload.
  stateRef.state = freshState(data.header);
  applyFn(stateRef.state, { t: 0, type: 'session_init', payload: data.header });
  renderFn(stateRef.state);

  setStatus(`live with ${data.header?.npc_name ?? npcId} — say something`);
  setComposerEnabled(true);
  document.getElementById('composer-input')?.focus();
}

// ---------------- network: /turn (SSE) --------------------------------------

async function onSendClick() {
  if (inFlight) return;
  if (!currentSessionId) { setStatus('start a session first'); return; }

  const input = document.getElementById('composer-input');
  const text = (input?.value ?? '').trim();
  if (!text) return;

  inFlight = true;
  setComposerEnabled(false);
  setComposerStatus('sending…');

  const headers = { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' };
  const key = localStorage.getItem(LS_KEY);
  if (key) headers['X-Gemini-Key'] = key;
  const body = JSON.stringify({
    session_id: currentSessionId,
    player_input: text,
    ...(key ? { gemini_key: key } : {}),
  });

  let res;
  try {
    res = await fetch(`${BACKEND_URL}/turn`, { method: 'POST', headers, body });
  } catch (err) {
    setComposerStatus('');
    setStatus(`network error: ${err.message}`);
    inFlight = false;
    setComposerEnabled(true);
    return;
  }

  if (res.status === 404) {
    setStatus('session not found — start a new one');
    clearPersistedSession();
    inFlight = false;
    setComposerEnabled(false);
    setComposerStatus('');
    return;
  }
  if (res.status === 410) {
    setStatus('session ended (cap reached) — start a new one');
    clearPersistedSession();
    inFlight = false;
    setComposerEnabled(false);
    setComposerStatus('');
    return;
  }
  if (res.status === 429) {
    const j = await safeJSON(res);
    setComposerStatus(`Rate limited — try again in ${j?.retry_after_s ?? '?'}s`);
    inFlight = false;
    setComposerEnabled(true);
    return;
  }
  if (res.status === 503) {
    setComposerStatus('server out of API quota — provide your key in Settings');
    inFlight = false;
    setComposerEnabled(true);
    return;
  }
  if (!res.ok || !res.body) {
    setComposerStatus(`turn failed: ${res.status}`);
    inFlight = false;
    setComposerEnabled(true);
    return;
  }

  // Clear input before stream starts (player line lands via turn_start event).
  if (input) { input.value = ''; input.style.height = 'auto'; }
  setComposerStatus('streaming…');

  await consumeSSE(res);

  setComposerStatus('');
  inFlight = false;
  setComposerEnabled(true);
  document.getElementById('composer-input')?.focus();
}

async function consumeSSE(res) {
  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let sawTurnEnd = false;

  while (true) {
    let chunk;
    try {
      chunk = await reader.read();
    } catch (err) {
      console.error('SSE read failed', err);
      break;
    }
    const { value, done } = chunk;
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line. Tolerate \n\n and \r\n\r\n.
    let idx;
    while ((idx = nextEventBoundary(buf)) !== -1) {
      const block = buf.slice(0, idx.start);
      buf = buf.slice(idx.end);
      const dataLines = block.split(/\r?\n/).filter(l => l.startsWith('data: '));
      if (!dataLines.length) continue;
      const json = dataLines.map(l => l.slice(6)).join('\n');
      try {
        const event = JSON.parse(json);
        applyFn(stateRef.state, event);
        renderFn(stateRef.state);
        if (event.type === 'turn_end') sawTurnEnd = true;
      } catch (e) {
        console.error('SSE parse', e, json);
      }
    }
  }

  // Flush trailing partial event (server ought to terminate with \n\n).
  if (buf.trim()) {
    const dataLines = buf.split(/\r?\n/).filter(l => l.startsWith('data: '));
    if (dataLines.length) {
      const json = dataLines.map(l => l.slice(6)).join('\n');
      try {
        const event = JSON.parse(json);
        applyFn(stateRef.state, event);
        renderFn(stateRef.state);
        if (event.type === 'turn_end') sawTurnEnd = true;
      } catch (_) { /* ignore */ }
    }
  }

  if (!sawTurnEnd) {
    // stream closed without a turn_end — show a hint but don't error
    setStatus('stream ended');
  }
}

/** Find the next SSE event boundary (blank line) and return start/end offsets. */
function nextEventBoundary(buf) {
  const a = buf.indexOf('\n\n');
  const b = buf.indexOf('\r\n\r\n');
  if (a === -1 && b === -1) return -1;
  if (a !== -1 && (b === -1 || a < b)) return { start: a, end: a + 2 };
  return { start: b, end: b + 4 };
}

// ---------------- helpers ---------------------------------------------------

function setComposerEnabled(enabled) {
  const send  = document.getElementById('composer-send');
  const input = document.getElementById('composer-input');
  if (send)  send.disabled  = !enabled;
  if (input) input.disabled = !enabled;
  const composer = document.getElementById('composer');
  if (composer) composer.classList.toggle('inert', !enabled);
}

function setStatus(msg) {
  const el = document.getElementById('status-text');
  if (el) el.textContent = msg;
}

function setComposerStatus(msg) {
  const el = document.getElementById('composer-status');
  if (el) el.textContent = msg;
}

function clearPersistedSession() {
  currentSessionId = null;
  localStorage.removeItem(LS_SESSION);
  localStorage.removeItem(LS_NPC);
}

async function safeJSON(res) {
  try { return await res.json(); } catch (_) { return null; }
}

function shortId(id) {
  return id && id.length > 10 ? id.slice(0, 8) + '…' : (id ?? '');
}
