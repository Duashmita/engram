/**
 * app.js, engram-chars entry point.
 *
 * Extends the engram replay/live system with a Three.js character viewport.
 * The character is driven by the same state/event stream as the memory panels.
 */

import { freshState, apply, rebuild } from './js/state.js';
import { initRadar, renderAll }        from './js/render.js';
import { enterLive, exitLive, startCustomSession, startPresetSession, injectNpcLine, getApiKey } from './js/live.js';
import { BACKEND_URL } from './config.js';
import { createCharacter }             from './js/character.js';
import { handleEvent, setInitialIdle } from './js/animations.js';
import { startOnboarding }             from './js/onboard.js';
import { showStartScreen, saveToCatalogue } from './js/catalogue.js';

// Built-in premade characters shown on the start screen.
const PRESET_ENTRIES = [
  { id: 'guard',    name: 'Rico, the Paranoid Guard', archetype: 'The Wary Sentinel',  source: 'preset', assetPath: 'assets/characters/guard',    ocean: { O:0.2, C:0.5, E:0.3, A:0.2, N:0.9 }, persona: 'A weathered dock guard who trusts no one.' },
  { id: 'merchant', name: 'Rico, the Friendly Merchant', archetype: 'The Warm Broker', source: 'preset', assetPath: 'assets/characters/merchant', ocean: { O:0.5, C:0.5, E:0.9, A:0.8, N:0.2 }, persona: 'A warm, talkative dockside trader.' },
  { id: 'clerk',    name: 'Rico, the Rigid Clerk',    archetype: 'The Rigid Archivist', source: 'preset', assetPath: 'assets/characters/clerk',    ocean: { O:0.1, C:0.9, E:0.3, A:0.5, N:0.4 }, persona: 'A by-the-book records clerk.' },
  { id: 'jeanie',   name: 'Jeanie',                   archetype: 'The Anxious Scholar', source: 'preset', assetPath: 'assets/characters/jeanie',   ocean: { O:0.65, C:0.85, E:0.4, A:0.65, N:0.85 }, persona: 'An anxious, driven MIT researcher.' },
  { id: 'maya',     name: 'Maya',                     archetype: 'The Open Wanderer',  source: 'preset', assetPath: 'assets/characters/maya',     ocean: { O:0.9, C:0.2, E:0.85, A:0.8, N:0.4 }, persona: 'A chaotic, free-spirited artist.' },
  { id: 'hale',     name: 'Inspector Hale',           archetype: 'The Blunt Detective', source: 'preset', assetPath: 'assets/characters/hale',     ocean: { O:0.4, C:0.85, E:0.3, A:0.2, N:0.25 }, persona: 'A blunt, exacting detective.' },
];

let startScreen = null;     // handle from showStartScreen
let previewChar = null;     // 3D character shown on the start screen

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

// Surface any uncaught error on-page so failures are visible without DevTools.
function showError(msg) {
  let b = document.getElementById('app-error-banner');
  if (!b) {
    b = document.createElement('div');
    b.id = 'app-error-banner';
    b.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:99999;background:#e23b54;color:#fff;font:13px/1.4 system-ui;padding:8px 14px;white-space:pre-wrap;';
    document.body.appendChild(b);
  }
  b.textContent = 'Error: ' + msg;
}
window.addEventListener('error', e => showError((e.message || 'script error') + (e.filename ? ' @ ' + e.filename.split('/').pop() + ':' + e.lineno : '')));
window.addEventListener('unhandledrejection', e => showError('promise: ' + (e.reason?.message || e.reason || 'unknown')));

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
    STATUS('Character failed to load, check console');
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

  // Unity export isn't live yet — route the click to the waitlist so interested
  // users can register their interest. Falls back to the toast if the dialog
  // module fails to load.
  document.getElementById('btn-unity')?.addEventListener('click', () => {
    import('./js/waitlist.js')
      .then(m => m.openWaitlist?.())
      .catch(() => showToast('Unity export coming soon'));
  });

  // Prepare Live-mode UI chrome (kept hidden behind the start/onboarding overlay).
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
  livePicker.classList.add('hidden');     // catalogue + New character button replace it
  composer.classList.remove('hidden');

  // Wire live plumbing once (composer, SSE, onSessionStart). resume:false so it
  // never auto-restores a stale session.
  enterLive({ stateRef, applyFn: apply, renderFn: renderAll, onSessionStart: initCharacter, resume: false });

  // Waitlist dialog (file owned by the catalogue layer; guard so a missing
  // module can never break boot).
  import('./js/waitlist.js').then(m => m.initWaitlist?.()).catch(() => {});

  // Show the catalogue start screen first.
  showStart();
}

// ── Start screen / catalogue ───────────────────────────────────────────────
function showStart() {
  try {
    document.body.classList.add('onboarding-active');   // hide the cockpit
    let mount = document.getElementById('start-mount');
    if (!mount) {
      mount = document.createElement('div');
      mount.id = 'start-mount';
      mount.className = 'start-mount';
      document.body.appendChild(mount);
    }
    mount.style.display = '';
    console.log('[start] showing catalogue with', PRESET_ENTRIES.length, 'presets');
    startScreen = showStartScreen({
      mountEl: mount,
      presets: PRESET_ENTRIES,
      onPreview: previewEntry,
      onPlay: playEntry,
      onCreateNew: () => { closeStart(); launchWizard(); },
    });
  } catch (e) {
    console.error('[start] showStart failed:', e);
    showError('start screen failed: ' + (e?.message || e));
  }
}

function closeStart() {
  if (previewChar) { previewChar.dispose?.(); previewChar = null; }
  if (startScreen) { startScreen.destroy?.(); startScreen = null; }
  const m = document.getElementById('start-mount');
  if (m) m.style.display = 'none';
}

// Load the selected character into the start-screen viewport (see their face).
// IMPORTANT: create the WebGL renderer ONCE per canvas, then only swap the model
// on each selection. Re-creating a WebGLRenderer on every click loses the WebGL
// context (browsers cap active contexts) and the viewport freezes after a few
// selections, which looks like "can't select".
async function previewEntry(entry, canvas) {
  if (!canvas) return;
  try {
    if (!previewChar || previewChar._canvas !== canvas) {
      if (previewChar) { previewChar.dispose?.(); previewChar = null; }
      previewChar = await createCharacter(canvas, null, { previewOnly: true });
      previewChar._canvas = canvas;
      previewChar.setBreathing?.(true);
      previewChar.lookAtPointer?.(true);
    }
    let url = null;
    if (entry.source === 'custom' && entry.glbUrl) {
      url = `${BACKEND_URL}/proxy_glb?url=${encodeURIComponent(entry.glbUrl)}`;
    } else if (entry.assetPath) {
      url = `${entry.assetPath}/base.glb`;
    }
    if (url && previewChar.loadModelFromUrl) await previewChar.loadModelFromUrl(url);
  } catch (e) { console.warn('[preview] failed', e); }
}

// Enter the cockpit and start a session for a premade or saved character.
async function playEntry(entry) {
  closeStart();
  document.body.classList.remove('onboarding-active');
  installNewCharacterButton();
  localStorage.removeItem('engram_live_session_id');
  localStorage.removeItem('engram_live_npc_id');
  try {
    let ok;
    if (entry.source === 'custom') {
      ok = await startCustomSession(entry);
      if (ok && entry.glbUrl && char?.loadModelFromUrl) {
        try { await char.loadModelFromUrl(`${BACKEND_URL}/proxy_glb?url=${encodeURIComponent(entry.glbUrl)}`); } catch (_) {}
      }
    } else {
      ok = await startPresetSession(entry.id, entry.ocean);
    }
    // The start functions surface their own failure status; without a live
    // session the birth moment (greeting) would be wrong, so skip it.
    if (!ok) return;
    await birthMoment(entry);
  } catch (e) {
    console.error('[play] failed:', e);
    STATUS('Error starting session, check console');
  }
}

function launchWizard() {
  document.body.classList.add('onboarding-active');
  startOnboarding({ onComplete: onWizardComplete });
}

async function onWizardComplete(characterConfig) {
  customCharacter = characterConfig;
  document.body.classList.remove('onboarding-active');
  localStorage.removeItem('engram_live_session_id');
  localStorage.removeItem('engram_live_npc_id');
  installNewCharacterButton();

  // Save the new character to the catalogue. Its 3D face (glbUrl) is filled in
  // later when background generation finishes.
  const slug = (characterConfig.name || 'custom')
    .toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '') || 'custom';
  const entry = {
    id: 'custom-' + slug,
    name: characterConfig.name,
    archetype: characterConfig.archetype || 'Custom',
    ocean: characterConfig.ocean,
    persona: characterConfig.persona,
    backstory: characterConfig.backstory || [],
    facts: characterConfig.facts || [],
    appearanceDescription: characterConfig.appearanceDescription || '',
    source: 'custom',
  };
  try { saveToCatalogue(entry); } catch (_) {}

  try {
    const ok = await startCustomSession(characterConfig);
    if (!ok) return;   // start failed; its status message is already shown
    if (!char) await initCharacter(slug);
    await birthMoment(characterConfig);
    if (characterConfig.genJobId) pollAndSwapModel(characterConfig.genJobId, entry);
  } catch (e) {
    console.error('[wizard] live launch failed:', e);
    STATUS('Error starting Live mode, check console');
  }
}

// Add a small "✦ New character" button to the topbar that re-runs onboarding.
function installNewCharacterButton() {
  if (document.getElementById('btn-new-char')) return;
  const btn = document.createElement('button');
  btn.id = 'btn-new-char';
  btn.className = 'ctrl ctrl-wide';
  btn.title = 'Create a new character';
  btn.innerHTML = '<span>New character</span>';
  // Return to the catalogue start screen rather than a full reload.
  btn.addEventListener('click', () => {
    if (char) { char.dispose?.(); char = null; }
    if (charRAF) { cancelAnimationFrame(charRAF); charRAF = null; }
    showStart();
  });
  const settings = document.getElementById('btn-settings');
  settings?.parentNode?.insertBefore(btn, settings);
}

// ── Progressive 3D build: poll the backend and hot-swap the model as Meshy
// finishes each stage (raw mesh → textured → rigged). The character starts as
// the grey placeholder and upgrades in place over a few minutes.
async function pollAndSwapModel(jobId, catalogueEntry) {
  let lastUrl = null;
  const STAGE_LABEL = { preview: 'sculpting', refine: 'texturing', rig: 'rigging' };
  for (let i = 0; i < 160; i++) {           // ~160 x 8s ~ 21 min cap
    await new Promise(r => setTimeout(r, 8000));
    let job;
    try {
      const res = await fetch(`${BACKEND_URL}/character_status/${jobId}`);
      if (!res.ok) break;
      job = await res.json();
    } catch (_) { continue; }
    if (!job || job.status === 'error') { break; }

    if (job.stage && STAGE_LABEL[job.stage]) {
      // The backend job has no name field; use the catalogue entry's.
      STATUS(`${catalogueEntry?.name ?? 'your character'}: ${STAGE_LABEL[job.stage]} (${job.progress ?? 0}%)`);
    }
    // Persist the generated face to the catalogue so it shows next time.
    if (job.glb_url && catalogueEntry) {
      try { saveToCatalogue({ ...catalogueEntry, glbUrl: job.glb_url }); } catch (_) {}
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

// ── The "birth" moment, the payoff after creation ────────────────────────────
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
    const key = getApiKey();   // migrates the legacy localStorage name forward
    const body = { name: config.name, persona: config.persona, ocean: config.ocean };
    // Let the backend record the greeting in this session's history so the
    // NPC knows it already spoke first.
    const sid = localStorage.getItem('engram_live_session_id');
    if (sid) body.session_id = sid;
    if (key) { body.anthropic_key = key; headers['X-Anthropic-Key'] = key; }
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
    document.getElementById('status-text').textContent = 'Boot error, open console (Cmd+Opt+I)';
  }
});
