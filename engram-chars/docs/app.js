/**
 * app.js — engram-chars entry point.
 *
 * Extends the engram replay/live system with a Three.js character viewport.
 * The character is driven by the same state/event stream as the memory panels.
 */

import { freshState, apply, rebuild } from './js/state.js';
import { initRadar, renderAll }        from './js/render.js';
import { enterLive, exitLive, startCustomSession, injectNpcLine } from './js/live.js';
import { BACKEND_URL } from './config.js';
import { createCharacter }             from './js/character.js';
import { handleEvent, setInitialIdle } from './js/animations.js';
import { startOnboarding }             from './js/onboard.js';

// Character config collected by the onboarding wizard (null until launch).
let customCharacter = null;

// ── Globals ──────────────────────────────────────────────────────────────────
let manifest    = null;
let activeEntry = null;
let events      = [];
let header      = null;
const stateRef  = { state: null };
function getState() { return stateRef.state; }
function setState(s) { stateRef.state = s; }

let i_emit      = 0;
let t_virtual   = 0;
let t_end       = 0;
let playing     = false;
let speed       = 1.0;
let raf_id      = null;
let last_frame_ts = null;

// Character
let char        = null;
let charRAF     = null;

const STATUS = (msg) => { document.getElementById('status-text').textContent = msg; };

// ── Mode ─────────────────────────────────────────────────────────────────────
const MODE_KEY = 'engram_mode';
let currentMode = 'live';  // start in live mode

// ── Character init ────────────────────────────────────────────────────────────
async function initCharacter(npcId) {
  const canvas = document.getElementById('char-canvas');
  if (!canvas) return;

  // Dispose previous character
  if (char) { char.dispose(); char = null; }
  if (charRAF) { cancelAnimationFrame(charRAF); charRAF = null; }

  // Check if this NPC has been generated before trying to load the GLB.
  // If not, go straight to the procedural placeholder (avoids noisy 404s).
  let assetPath = null;
  if (npcId) {
    try {
      const res = await fetch(`assets/characters/${npcId}/manifest.json`);
      if (res.ok) assetPath = `assets/characters/${npcId}`;
    } catch (_) {}
  }

  try {
    char = await createCharacter(canvas, assetPath);
  } catch (e) {
    console.error('[app] createCharacter failed:', e);
    STATUS('Character failed to load — check console');
    return;
  }
  console.log('[app] character created', { npcId, assetPath, hasChar: !!char,
    w: canvas.clientWidth, h: canvas.clientHeight });
  STATUS(assetPath ? 'Character loaded' : 'Using placeholder character (no 3D model generated for this NPC yet)');

  // createCharacter self-drives its own render loop now; update() is a no-op,
  // but we keep a light loop for any state-driven hooks.
  function animLoop() {
    charRAF = requestAnimationFrame(animLoop);
    if (char) char.update();
  }
  animLoop();

  // Apply initial idle based on current state
  if (getState()) setInitialIdle(getState(), char);
}

// ── Render wrapper that also updates the character ────────────────────────────
let _prevState = null;
function renderWithChar(state) {
  renderAll(state);
  if (char && state) {
    handleEvent(state, _prevState, char);
  }
  _prevState = state ? { ...state } : null;
}

// ── Replay mode ───────────────────────────────────────────────────────────────
function resetReplay() {
  if (raf_id) { cancelAnimationFrame(raf_id); raf_id = null; }
  i_emit    = 0;
  t_virtual = 0;
  playing   = false;
  last_frame_ts = null;
  if (header) { setState(freshState(header)); renderWithChar(getState()); }
  document.getElementById('btn-play').textContent = '▶';
  scrubberSync();
}

function scrubberSync() {
  const el = document.getElementById('scrubber');
  el.max   = Math.round(t_end * 1000);
  el.value = Math.round(t_virtual * 1000);
  document.getElementById('time-cur').textContent = t_virtual.toFixed(2) + 's';
  document.getElementById('time-end').textContent = t_end.toFixed(2) + 's';
  document.getElementById('event-counter').textContent = `${i_emit}/${events.length} events`;
}

function applyUpTo(t) {
  while (i_emit < events.length && events[i_emit].t <= t) {
    apply(getState(), events[i_emit]);
    i_emit++;
  }
}

function tick(ts) {
  if (!playing) return;
  if (last_frame_ts !== null) {
    const dt = Math.min((ts - last_frame_ts) / 1000, 0.1) * speed;
    t_virtual = Math.min(t_virtual + dt, t_end);
    applyUpTo(t_virtual);
    renderWithChar(getState());
    scrubberSync();
    if (t_virtual >= t_end) {
      playing = false;
      document.getElementById('btn-play').textContent = '▶';
    }
  }
  last_frame_ts = ts;
  if (playing) raf_id = requestAnimationFrame(tick);
}

function stepOne() {
  if (i_emit < events.length) {
    apply(getState(), events[i_emit]);
    t_virtual = events[i_emit].t;
    i_emit++;
    renderWithChar(getState());
    scrubberSync();
  }
}

function stepToNextTurn() {
  const startTurn = getState().turns?.length || 0;
  while (i_emit < events.length) {
    apply(getState(), events[i_emit]);
    t_virtual = events[i_emit].t;
    i_emit++;
    const newTurn = getState().turns?.length || 0;
    if (newTurn > startTurn) break;
  }
  renderWithChar(getState());
  scrubberSync();
}

// ── Session file loading ──────────────────────────────────────────────────────
async function loadSession(entry) {
  activeEntry = entry;
  events = [];
  STATUS(`Loading ${entry.file}…`);

  const res = await fetch(`sessions/${entry.file}`);
  const text = await res.text();

  for (const line of text.split('\n')) {
    const t = line.trim();
    if (!t) continue;
    try {
      const ev = JSON.parse(t);
      if (ev.type === 'session_init' || ev.type === 'session_init_npc') {
        header = ev.payload;
      } else {
        events.push(ev);
      }
    } catch (_) {}
  }

  t_end = events.length > 0 ? events[events.length - 1].t : 0;
  setState(freshState(header));
  renderWithChar(getState());
  resetReplay();

  document.getElementById('group-badge').classList.toggle('hidden', !entry.group);
  if (entry.group) document.getElementById('group-badge').textContent = entry.group;

  STATUS('Session loaded');
  await initCharacter(entry.npc_id || header?.npc_id);
}

// ── Manifest loading ──────────────────────────────────────────────────────────
async function loadManifest() {
  try {
    const res = await fetch('sessions/manifest.json');
    manifest = await res.json();
  } catch (_) {
    manifest = { sessions: [] };
  }

  const sel = document.getElementById('session-select');
  sel.innerHTML = '';
  for (const s of manifest.sessions || []) {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = `${s.npc_name} (${s.id.slice(-8)})`;
    sel.appendChild(opt);
  }
  if (manifest.sessions?.length) loadSession(manifest.sessions[0]);
}

// ── Controls wiring ───────────────────────────────────────────────────────────
function boot() {
  initRadar(document.getElementById('ocean-radar'));

  // Transport
  document.getElementById('btn-play').addEventListener('click', () => {
    if (currentMode !== 'replay') return;
    playing = !playing;
    document.getElementById('btn-play').textContent = playing ? '⏸' : '▶';
    if (playing) { last_frame_ts = null; raf_id = requestAnimationFrame(tick); }
  });
  document.getElementById('btn-reset').addEventListener('click', resetReplay);
  document.getElementById('btn-step').addEventListener('click', stepOne);
  document.getElementById('btn-step-turn').addEventListener('click', stepToNextTurn);

  // Speed
  document.querySelectorAll('.speed button').forEach(btn => {
    btn.addEventListener('click', () => {
      speed = parseFloat(btn.dataset.speed);
      document.querySelectorAll('.speed button').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // Scrubber
  const scrubEl = document.getElementById('scrubber');
  scrubEl.addEventListener('input', () => {
    playing = false;
    document.getElementById('btn-play').textContent = '▶';
    t_virtual = parseInt(scrubEl.value) / 1000;
    setState(rebuild(events, t_virtual, header));
    renderWithChar(getState());
    i_emit = events.filter(e => e.t <= t_virtual).length;
    scrubberSync();
  });

  // Session picker
  document.getElementById('session-select').addEventListener('change', e => {
    const entry = manifest?.sessions?.find(s => s.id === e.target.value);
    if (entry) loadSession(entry);
  });

  // Mode toggle
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.mode;
      if (mode === currentMode) return;
      currentMode = mode;
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === mode));

      const replayControls = document.querySelector('.controls');
      const scrubRow = document.querySelector('.scrubber-row');
      const sessionPicker = document.querySelector('.session-picker');
      const livePicker = document.getElementById('live-picker');
      const composer = document.getElementById('composer');

      if (mode === 'live') {
        replayControls.style.display = 'none';
        scrubRow.style.display = 'none';
        sessionPicker.classList.add('hidden');
        livePicker.classList.remove('hidden');
        composer.classList.remove('hidden');
        enterLive({ stateRef, applyFn: apply, renderFn: renderAll, onSessionStart: initCharacter });
      } else {
        replayControls.style.display = '';
        scrubRow.style.display = '';
        sessionPicker.classList.remove('hidden');
        livePicker.classList.add('hidden');
        composer.classList.add('hidden');
        exitLive();
        loadManifest();
      }
    });
  });

  // Settings
  document.getElementById('btn-settings').addEventListener('click', () => {
    document.getElementById('settings-dialog').showModal();
  });

  // Export to Unity (placeholder — no functionality yet).
  document.getElementById('btn-unity')?.addEventListener('click', () => {
    showToast('Unity export coming soon');
  });

  // Prepare Live-mode UI chrome (kept hidden behind the onboarding overlay).
  const replayControls = document.querySelector('.controls');
  const scrubRow = document.querySelector('.scrubber-row');
  const sessionPicker = document.querySelector('.session-picker');
  const livePicker = document.getElementById('live-picker');
  const composer = document.getElementById('composer');

  document.querySelectorAll('.mode-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.mode === 'live')
  );
  replayControls.style.display = 'none';
  scrubRow.style.display = 'none';
  sessionPicker.classList.add('hidden');
  livePicker.classList.remove('hidden');
  composer.classList.remove('hidden');

  // Hide the main engram UI until the wizard launches the character.
  document.body.classList.add('onboarding-active');

  // Run the onboarding wizard first; reveal + start the live session on finish.
  startOnboarding({
    onComplete: async (characterConfig) => {
      customCharacter = characterConfig;
      document.body.classList.remove('onboarding-active');

      // Clear any stale preset session so enterLive doesn't resume "Jeanie".
      localStorage.removeItem('engram_live_session_id');
      localStorage.removeItem('engram_live_npc_id');

      // startCustomSession registers the character in the topbar dropdown and
      // selects it, so the picker stays visible and shows the new name. Add a
      // "New character" affordance too.
      installNewCharacterButton();

      try {
        // resume:false → don't restore a stale session or pop the OCEAN dialog.
        enterLive({ stateRef, applyFn: apply, renderFn: renderAll, onSessionStart: initCharacter, resume: false });
        // startCustomSession awaits initCharacter, so `char` is ready after it.
        await startCustomSession(characterConfig);
        // Safety net: if the session wired up but the model didn't load, force it.
        if (!char) {
          const slug = (characterConfig.name || 'custom')
            .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'custom';
          await initCharacter(slug);
        }
        await birthMoment(characterConfig);
        // If a background 3D build is running, progressively swap the model in.
        if (characterConfig.genJobId) pollAndSwapModel(characterConfig.genJobId);
      } catch (e) {
        console.error('[boot] live launch failed:', e);
        STATUS('Error starting Live mode — check console');
      }
    },
  });
}

// Add a small "✦ New character" button to the topbar that re-runs onboarding.
function installNewCharacterButton() {
  if (document.getElementById('btn-new-char')) return;
  const btn = document.createElement('button');
  btn.id = 'btn-new-char';
  btn.className = 'ctrl ctrl-wide';
  btn.title = 'Create a new character';
  btn.innerHTML = '<span>✦ New character</span>';
  btn.addEventListener('click', () => location.reload());
  const settings = document.getElementById('btn-settings');
  settings?.parentNode?.insertBefore(btn, settings);
}

// ── Progressive 3D build: poll the backend and hot-swap the model as Meshy
// finishes each stage (raw mesh → textured → rigged). The character starts as
// the grey placeholder and upgrades in place over a few minutes.
async function pollAndSwapModel(jobId) {
  let lastUrl = null;
  const STAGE_LABEL = { preview: 'sculpting…', refine: 'texturing…', rig: 'rigging…' };
  for (let i = 0; i < 160; i++) {           // ~160 × 8s ≈ 21 min cap
    await new Promise(r => setTimeout(r, 8000));
    let job;
    try {
      const res = await fetch(`${BACKEND_URL}/character_status/${jobId}`);
      if (!res.ok) break;
      job = await res.json();
    } catch (_) { continue; }
    if (!job || job.status === 'error') { break; }

    if (job.stage && STAGE_LABEL[job.stage]) {
      STATUS(`✨ ${job.name ?? 'your character'} — ${STAGE_LABEL[job.stage]} (${job.progress ?? 0}%)`);
    }
    // A newer GLB is available → swap it into the viewport. Load via our
    // backend proxy so the browser doesn't hit CORS on Meshy's CDN.
    if (job.glb_url && job.glb_url !== lastUrl && char?.loadModelFromUrl) {
      lastUrl = job.glb_url;
      const proxied = `${BACKEND_URL}/proxy_glb?url=${encodeURIComponent(job.glb_url)}`;
      try { await char.loadModelFromUrl(proxied); STATUS('✨ character updated'); }
      catch (e) { console.warn('[app] model swap failed', e); }
    }
    if (job.status === 'done') { STATUS('Character ready'); break; }
  }
}

// ── The "birth" moment — the payoff after creation ────────────────────────────
// The character materializes, waves, and speaks first (unprompted), turning the
// thing the user just built into a presence that greets them.
async function birthMoment(config) {
  if (char) {
    char.setBreathing?.(true);
    char.lookAtPointer?.(true);
    try { await char.materialize?.(1200); } catch (_) {}
    char.greet?.();
  }
  // Fetch and show the character's first spoken line.
  try {
    const headers = { 'Content-Type': 'application/json' };
    const key = localStorage.getItem('engram_gemini_key');
    const body = { name: config.name, persona: config.persona, ocean: config.ocean };
    if (key) body.anthropic_key = key;
    const res = await fetch(`${BACKEND_URL}/greeting`, {
      method: 'POST', headers, body: JSON.stringify(body),
    });
    if (res.ok) {
      const { greeting } = await res.json();
      if (greeting) {
        // Small beat so the line lands after the wave begins.
        setTimeout(() => injectNpcLine(greeting), 600);
      }
    }
  } catch (e) {
    console.warn('[birth] greeting failed', e);
  }
}

function showToast(msg) {
  const t = document.createElement('div');
  t.className = 'app-toast';
  t.textContent = msg;
  document.body.appendChild(t);
  requestAnimationFrame(() => t.classList.add('visible'));
  setTimeout(() => {
    t.classList.remove('visible');
    setTimeout(() => t.remove(), 250);
  }, 2200);
}

document.addEventListener('DOMContentLoaded', () => {
  try {
    boot();
  } catch (e) {
    console.error('[app] boot failed:', e);
    document.getElementById('status-text').textContent = 'Boot error — open console (Cmd+Opt+I)';
  }
});
