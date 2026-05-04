// app.js — Engram Memory Replay (entry point)
//
// Architecture:
//   * Manifest at sessions/manifest.json lists available sessions.
//   * Each session is an NDJSON file under sessions/.
//   * Loading a session parses all events, finds the session_init header,
//     and renders the initial empty state.
//   * Playback loop is RAF-driven: a virtual clock t_virtual advances by
//     dt * speed, and any unconsumed event with t <= t_virtual is applied
//     to state, then the renderer is called once per frame.
//   * Scrubbing: snap t_virtual, then rebuild state from events 0..i where
//     i is the last event with t <= t_virtual. Render once.
//
// State is in state.js (pure). DOM updates in render.js (idempotent).
// This file owns the *control flow*: timeline, controls, manifest plumbing.

import { freshState, apply, rebuild } from './js/state.js';
import { initRadar, renderAll } from './js/render.js';
import { enterLive, exitLive } from './js/live.js';

// --------------- globals (single-NPC mode) ---------------
let manifest    = null;        // {sessions: [...]}
let activeEntry = null;        // currently loaded manifest entry
let events      = [];          // event array for active session
let header      = null;        // session_init payload
// `state` is held inside `stateRef` so live.js can swap the underlying object
// (e.g. on /start) and we both observe the change.
const stateRef  = { state: null };
function getState() { return stateRef.state; }
function setState(s) { stateRef.state = s; }

let i_emit      = 0;           // index of next unconsumed event
let t_virtual   = 0;           // virtual playback clock (seconds)
let t_end       = 0;           // last event t
let playing     = false;
let speed       = 1.0;
let raf_id      = null;
let last_frame_ts = null;

const STATUS = (msg) => { document.getElementById('status-text').textContent = msg; };

// --------------- boot ---------------
const MODE_KEY = 'engram_mode';
let currentMode = 'replay';

async function boot() {
  // chart (shared across modes)
  initRadar(document.getElementById('ocean-radar'));

  // wire replay controls
  document.getElementById('btn-play').addEventListener('click', togglePlay);
  document.getElementById('btn-step').addEventListener('click', stepOne);
  document.getElementById('btn-step-turn').addEventListener('click', stepTurn);
  document.getElementById('btn-reset').addEventListener('click', reset);
  document.querySelectorAll('.speed button').forEach(b =>
    b.addEventListener('click', () => setSpeed(parseFloat(b.dataset.speed), b))
  );
  document.getElementById('scrubber').addEventListener('input', onScrub);
  document.getElementById('session-select').addEventListener('change', onSelectSession);

  // wire mode toggle
  document.querySelectorAll('.mode-btn').forEach(b =>
    b.addEventListener('click', () => switchMode(b.dataset.mode))
  );

  // pick mode from localStorage (default replay)
  const initial = localStorage.getItem(MODE_KEY) === 'replay' ? 'replay' : 'live';
  await switchMode(initial, /*persist=*/false);
}

async function switchMode(mode, persist = true) {
  if (mode !== 'replay' && mode !== 'live') mode = 'replay';
  currentMode = mode;
  if (persist) localStorage.setItem(MODE_KEY, mode);

  // toggle button highlight
  document.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === mode)
  );
  // brand subtitle reflects mode
  const sub = document.getElementById('brand-sub');
  if (sub) sub.textContent = mode === 'live' ? 'Live Chat' : 'Memory Replay';

  if (mode === 'replay') {
    exitLive();
    await loadManifest();
  } else {
    pause();
    enterLive({ stateRef, applyFn: apply, renderFn: renderAll });
  }
}

async function loadManifest() {
  // load manifest (same as legacy boot tail)
  try {
    const res = await fetch('sessions/manifest.json', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`manifest ${res.status}`);
    manifest = await res.json();
  } catch (err) {
    STATUS(`could not load manifest: ${err.message}`);
    manifest = { sessions: [] };
  }

  populateSessionSelector();

  if (manifest.sessions.length === 0) {
    STATUS('no sessions in manifest');
    return;
  }

  // default-select first
  await loadSession(manifest.sessions[0]);
}

function populateSessionSelector() {
  const sel = document.getElementById('session-select');
  sel.innerHTML = '';

  // group by .group, ungrouped first
  const ungrouped = [];
  const groups = new Map();
  for (const s of manifest.sessions) {
    if (!s.group) ungrouped.push(s);
    else {
      if (!groups.has(s.group)) groups.set(s.group, []);
      groups.get(s.group).push(s);
    }
  }

  for (const s of ungrouped) {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = `${s.npc_name ?? s.npc_id ?? '?'} — ${s.id}`;
    sel.appendChild(opt);
  }
  for (const [name, list] of groups) {
    const og = document.createElement('optgroup');
    og.label = `Group: ${name}`;
    for (const s of list) {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = `${s.npc_name ?? s.npc_id ?? '?'} — ${s.id}`;
      og.appendChild(opt);
    }
    sel.appendChild(og);
  }
}

async function onSelectSession() {
  const sel = document.getElementById('session-select');
  const entry = manifest.sessions.find(s => s.id === sel.value);
  if (entry) await loadSession(entry);
}

// --------------- session loading ---------------
async function loadSession(entry) {
  pause();
  activeEntry = entry;
  STATUS(`loading ${entry.file}…`);

  let text;
  try {
    const res = await fetch(`sessions/${entry.file}`, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    text = await res.text();
  } catch (err) {
    STATUS(`failed: ${err.message}`);
    return;
  }

  // parse NDJSON, tolerate \r\n and blank lines
  events = [];
  for (const raw of text.split('\n')) {
    const line = raw.replace(/\r$/, '').trim();
    if (!line) continue;
    try { events.push(JSON.parse(line)); }
    catch (e) { console.warn('bad NDJSON line skipped:', line.slice(0, 80)); }
  }

  if (events.length === 0) {
    STATUS('session is empty');
    return;
  }

  // first event must be session_init (or session_init_npc)
  const first = events[0];
  if (first.type !== 'session_init' && first.type !== 'session_init_npc') {
    STATUS('warning: first event is not session_init');
  }
  header = first.payload ?? {};
  t_end  = events[events.length - 1].t ?? 0;

  // group badge
  const badge = document.getElementById('group-badge');
  if (entry.group) {
    badge.textContent = `Group: ${entry.group}`;
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }

  // scrubber range
  const sc = document.getElementById('scrubber');
  sc.max = String(Math.max(1, Math.ceil(t_end * 1000)));
  sc.value = '0';

  document.getElementById('time-end').textContent = `${t_end.toFixed(2)}s`;
  document.getElementById('event-counter').textContent = `0/${events.length}`;

  reset();
  STATUS(`loaded ${events.length} events`);
}

// --------------- replay engine ---------------
function reset() {
  pause();
  setState(freshState(header));
  // session_init has been consumed conceptually — but we still want to apply it
  // to overwrite freshState defaults if anything diverges. Index advances past it.
  i_emit = 0;
  t_virtual = 0;
  // apply session_init if present at index 0 (no-op for clean state but safe)
  if (events[0] && (events[0].type === 'session_init' || events[0].type === 'session_init_npc')) {
    apply(getState(), events[0]);
    i_emit = 1;
  }
  updateClockUI();
  renderAll(getState());
}

function togglePlay() {
  if (playing) pause(); else play();
}

function play() {
  if (i_emit >= events.length) {
    // restart from beginning if we're at the end
    reset();
  }
  playing = true;
  document.getElementById('btn-play').textContent = '❚❚';
  last_frame_ts = null;
  raf_id = requestAnimationFrame(tick);
}

function pause() {
  playing = false;
  document.getElementById('btn-play').textContent = '▶';
  if (raf_id != null) cancelAnimationFrame(raf_id);
  raf_id = null;
}

function tick(ts) {
  if (!playing) return;
  if (last_frame_ts == null) last_frame_ts = ts;
  const dt = (ts - last_frame_ts) / 1000.0;
  last_frame_ts = ts;
  t_virtual += dt * speed;

  // emit any events whose t <= t_virtual
  let advanced = false;
  while (i_emit < events.length && (events[i_emit].t ?? 0) <= t_virtual) {
    apply(getState(), events[i_emit]);
    i_emit++;
    advanced = true;
  }

  if (advanced) renderAll(getState());
  updateClockUI();

  if (i_emit >= events.length) {
    pause();
    STATUS('session complete');
    return;
  }
  raf_id = requestAnimationFrame(tick);
}

function stepOne() {
  pause();
  if (i_emit >= events.length) return;
  apply(getState(), events[i_emit]);
  t_virtual = events[i_emit].t ?? t_virtual;
  i_emit++;
  renderAll(getState());
  updateClockUI();
}

function stepTurn() {
  pause();
  while (i_emit < events.length) {
    const ev = events[i_emit];
    apply(getState(), ev);
    t_virtual = ev.t ?? t_virtual;
    i_emit++;
    if (ev.type === 'turn_end') break;
  }
  renderAll(getState());
  updateClockUI();
}

function onScrub(e) {
  pause();
  const target = parseFloat(e.target.value) / 1000.0;  // ms → s
  t_virtual = target;
  // rebuild deterministically up to the latest event with t <= target
  let upto = -1;
  for (let k = 0; k < events.length; k++) {
    if ((events[k].t ?? 0) <= target) upto = k; else break;
  }
  setState(rebuild(events, upto, header));
  i_emit = upto + 1;
  renderAll(getState());
  updateClockUI();
}

function setSpeed(v, btn) {
  speed = v;
  document.querySelectorAll('.speed button').forEach(b => b.classList.toggle('active', b === btn));
}

function updateClockUI() {
  document.getElementById('time-cur').textContent = `${t_virtual.toFixed(2)}s`;
  document.getElementById('event-counter').textContent = `${i_emit}/${events.length}`;
  const sc = document.getElementById('scrubber');
  // only update slider if user isn't dragging (avoid feedback loop)
  if (document.activeElement !== sc) {
    sc.value = String(Math.round(t_virtual * 1000));
  }
}

// --------------- go ---------------
boot().catch(err => {
  console.error(err);
  STATUS(`boot failed: ${err.message}`);
});
