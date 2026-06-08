// onboard.js — Character-creation as a rewarding journey.
//
// The arc: name → interview (personality forms in real time) → the reveal
// (archetype + radar draws itself) → taking shape (appearance types out) →
// a past (memories fly into them) → birth (handed back to app.js, where the
// character materializes, waves, and speaks first).
//
// The grey 3D caricature is alive throughout — breathing, watching the cursor.
// Visual reward components come from onboard-visuals.js.

import { BACKEND_URL } from '../config.js';
import { createCharacter } from './character.js';
import {
  createFormingRadar, typewriter, revealArchetype,
  flyInMemoryCard, buildCharacterCard, confetti,
} from './onboard-visuals.js';

const LS_KEY = 'engram_gemini_key';

const QUESTIONS = [
  { trait: 'O', tag: 'their curiosity',   q: 'When your character meets something new or unfamiliar — do they lean in, or hold back?' },
  { trait: 'C', tag: 'their discipline',  q: 'How do they handle their work, their plans, their promises?' },
  { trait: 'E', tag: 'their energy',      q: 'Drop them into a room full of strangers. What do they do?' },
  { trait: 'A', tag: 'their warmth',      q: 'Someone challenges them, flatly disagrees. How do they take it?' },
  { trait: 'N', tag: 'their composure',   q: 'Things go wrong. Pressure mounts. How do they hold up?' },
];

const TRAITS = [
  { key: 'O', label: 'Openness' },
  { key: 'C', label: 'Conscientiousness' },
  { key: 'E', label: 'Extraversion' },
  { key: 'A', label: 'Agreeableness' },
  { key: 'N', label: 'Neuroticism' },
];

function apiKey() {
  const k = localStorage.getItem(LS_KEY);
  return k && k.trim() ? k.trim() : null;
}
function postJSON(path, bodyObj) {
  const headers = { 'Content-Type': 'application/json' };
  const key = apiKey();
  if (key) bodyObj.anthropic_key = key;
  return fetch(`${BACKEND_URL}${path}`, {
    method: 'POST', headers, body: JSON.stringify(bodyObj),
  });
}

const data = {
  name: '',
  answers: [],
  ocean: { O: 0.5, C: 0.5, E: 0.5, A: 0.5, N: 0.5 },
  summary: '',
  archetype: '',
  appearanceDescription: '',
  memories: '',
  notes: '',
};

export function startOnboarding({ onComplete }) {
  const overlay = document.createElement('div');
  overlay.id = 'onboard-overlay';
  overlay.className = 'onboard-overlay';
  overlay.innerHTML = `
    <div class="onboard-stage">
      <div class="onboard-char" id="onboard-char">
        <canvas id="onboard-canvas"></canvas>
        <div class="onboard-nameplate" id="onboard-nameplate"></div>
      </div>
      <div class="onboard-right">
        <div class="onboard-steps" id="onboard-steps"></div>
        <div class="onboard-radar-wrap" id="onboard-radar-wrap">
          <canvas id="onboard-radar"></canvas>
        </div>
        <div class="onboard-card" id="onboard-card"></div>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const card = overlay.querySelector('#onboard-card');
  const stepsEl = overlay.querySelector('#onboard-steps');
  const radarWrap = overlay.querySelector('#onboard-radar-wrap');
  const charEl = overlay.querySelector('#onboard-char');
  const nameplate = overlay.querySelector('#onboard-nameplate');

  // Smooth per-step entrance: re-trigger the fade/slide animation whenever the
  // card's direct children change (i.e. a new step renders). Grandchild
  // mutations (typewriter, radar reveal, slider edits) don't fire it.
  const cardObserver = new MutationObserver(() => {
    card.classList.remove('card-enter');
    void card.offsetWidth;       // force reflow so the animation restarts
    card.classList.add('card-enter');
  });
  cardObserver.observe(card, { childList: true });

  // ── Living grey caricature ─────────────────────────────────────────────────
  let char = null;
  createCharacter(overlay.querySelector('#onboard-canvas'), null)
    .then(c => {
      char = c;
      char.setBreathing?.(true);
      char.lookAtPointer?.(true);
      (function loop() {
        if (!document.getElementById('onboard-overlay')) { char.dispose?.(); return; }
        requestAnimationFrame(loop);
        char.update?.();
      })();
    })
    .catch(err => console.warn('[onboard] caricature failed', err));

  // ── Persistent forming radar (hidden until the interview) ──────────────────
  let radar = null;
  function ensureRadar() {
    if (!radar) radar = createFormingRadar(overlay.querySelector('#onboard-radar'));
    radarWrap.classList.add('visible');
    return radar;
  }

  // ── Journey progress constellation ─────────────────────────────────────────
  const STEP_COUNT = 5; // name, interview, reveal, appearance, memories
  function setStep(active) {
    stepsEl.innerHTML = Array.from({ length: STEP_COUNT }, (_, i) =>
      `<span class="onboard-dot ${i < active ? 'done' : ''} ${i === active ? 'on' : ''}"></span>`
    ).join('');
  }

  // ── 0. Threshold — name ─────────────────────────────────────────────────────
  function stepName() {
    setStep(0);
    radarWrap.classList.remove('visible');
    card.innerHTML = `
      <p class="onboard-kicker">Every character begins as a blank slate.</p>
      <h2 class="onboard-h">Name your character</h2>
      <input id="ob-name" class="onboard-input" type="text" placeholder="Give them a name…"
             autocomplete="off" value="${esc(data.name)}" />
      <div class="onboard-actions">
        <button id="ob-next" class="onboard-btn primary" disabled>Begin</button>
      </div>
    `;
    const input = card.querySelector('#ob-name');
    const next = card.querySelector('#ob-next');
    const sync = () => { next.disabled = !input.value.trim(); };
    input.addEventListener('input', sync);
    input.addEventListener('keydown', e => { if (e.key === 'Enter' && input.value.trim()) next.click(); });
    next.addEventListener('click', () => {
      data.name = input.value.trim();
      nameplate.textContent = data.name;
      nameplate.classList.add('visible');
      char?.greet?.();          // the figure acknowledges its new name
      stepInterview(0);
    });
    sync();
    setTimeout(() => input.focus(), 40);
  }

  // ── 1. Interview — personality forms as you answer ──────────────────────────
  function stepInterview(idx) {
    setStep(1);
    ensureRadar().setProgress(idx);
    if (idx >= QUESTIONS.length) { reveal(); return; }
    const item = QUESTIONS[idx];
    const prev = data.answers[idx]?.answer ?? '';
    card.innerHTML = `
      <p class="onboard-kicker">${idx + 1} of ${QUESTIONS.length} · ${esc(item.tag)}</p>
      <h2 class="onboard-h onboard-q">${esc(item.q)}</h2>
      <textarea id="ob-ans" class="onboard-textarea" rows="3"
                placeholder="A sentence or two is plenty…">${esc(prev)}</textarea>
      <div class="onboard-actions">
        ${idx > 0 ? '<button id="ob-back" class="onboard-btn ghost">Back</button>' : ''}
        <button id="ob-next" class="onboard-btn primary" disabled>${idx === QUESTIONS.length - 1 ? 'See who they are' : 'Next'}</button>
      </div>
    `;
    const ta = card.querySelector('#ob-ans');
    const next = card.querySelector('#ob-next');
    const sync = () => { next.disabled = !ta.value.trim(); };
    ta.addEventListener('input', sync);
    ta.addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && ta.value.trim()) next.click();
    });
    next.addEventListener('click', () => {
      data.answers[idx] = { question: item.q, answer: ta.value.trim() };
      ensureRadar().setProgress(idx + 1);   // the shape grows with this answer
      stepInterview(idx + 1);
    });
    card.querySelector('#ob-back')?.addEventListener('click', () => stepInterview(idx - 1));
    sync();
    setTimeout(() => ta.focus(), 40);
  }

  // ── 2. The reveal — archetype + radar draws itself ──────────────────────────
  async function reveal() {
    setStep(2);
    card.innerHTML = `
      <div class="onboard-loading">
        <div class="onboard-spinner"></div>
        <p class="onboard-loading-text">Reading ${esc(data.name)}…</p>
      </div>`;
    try {
      const res = await postJSON('/infer_ocean', { qa: data.answers });
      if (!res.ok) throw new Error(res.status);
      const j = await res.json();
      if (j.ocean) data.ocean = normalizeOcean(j.ocean);
      data.summary = j.summary || '';
      data.archetype = j.archetype || 'The Enigma';
    } catch (err) {
      console.warn('[onboard] infer_ocean error', err);
      data.summary = 'The personality model was unreachable — start from neutral and shape them by hand.';
      data.archetype = 'The Unknown';
    }

    // Archetype lands, then the radar draws to the true shape.
    card.innerHTML = `
      <div class="onboard-reveal" id="ob-reveal"></div>
      <div class="onboard-refine" id="ob-refine" style="opacity:0"></div>
    `;
    ensureRadar();
    await Promise.all([
      revealArchetype(card.querySelector('#ob-reveal'), {
        archetype: data.archetype, summary: data.summary,
      }),
      radar.reveal(data.ocean),
    ]);

    // Offer refinement (sliders) below the reveal, gently faded in.
    const refine = card.querySelector('#ob-refine');
    refine.innerHTML = `
      <p class="onboard-subnote">This is who your words made. Refine them, or leave them be.</p>
      <div class="onboard-sliders">
        ${TRAITS.map(t => `
          <div class="onboard-slider-row">
            <label for="ob-s-${t.key}"><b>${t.key}</b> ${t.label}</label>
            <input id="ob-s-${t.key}" type="range" min="0" max="100"
                   value="${Math.round((data.ocean[t.key] ?? .5) * 100)}" />
            <span class="onboard-slider-val" id="ob-v-${t.key}">${(data.ocean[t.key] ?? .5).toFixed(2)}</span>
          </div>`).join('')}
      </div>
      <div class="onboard-actions">
        <button id="ob-next" class="onboard-btn primary">Give them a face</button>
      </div>`;
    for (const t of TRAITS) {
      const s = refine.querySelector(`#ob-s-${t.key}`);
      const v = refine.querySelector(`#ob-v-${t.key}`);
      s.addEventListener('input', () => {
        const f = Math.round(parseFloat(s.value)) / 100;
        data.ocean[t.key] = f;
        v.textContent = f.toFixed(2);
        radar.reveal(data.ocean);    // radar tracks live edits
      });
    }
    refine.querySelector('#ob-next').addEventListener('click', stepAppearance);
    requestAnimationFrame(() => { refine.style.transition = 'opacity .6s ease'; refine.style.opacity = '1'; });
  }

  // ── 3. Taking shape — appearance types itself out ───────────────────────────
  async function stepAppearance() {
    setStep(3);
    card.innerHTML = `
      <h2 class="onboard-h">${esc(data.name)} takes shape</h2>
      <p class="onboard-subnote">Sculpted from these words. Edit anything — it's theirs.</p>
      <div class="onboard-appear" id="ob-appear"></div>
      <div class="onboard-actions">
        <button id="ob-regen" class="onboard-btn ghost">Regenerate</button>
        <button id="ob-next" class="onboard-btn primary">Give them a past</button>
      </div>`;
    const appear = card.querySelector('#ob-appear');

    async function load() {
      appear.innerHTML = `<div class="onboard-typing" id="ob-type"></div>`;
      char?.materialize?.(1100);   // the form shimmers in as the words arrive
      let text = data.appearanceDescription;
      if (!text) {
        try {
          const res = await postJSON('/appearance', {
            name: data.name, persona: data.summary, ocean: { ...data.ocean },
          });
          if (res.ok) text = (await res.json()).description || '';
        } catch (err) { console.warn('[onboard] appearance error', err); }
      }
      if (!text) text = 'Describe your character here — they will be sculpted from it.';
      data.appearanceDescription = text;
      await typewriter(card.querySelector('#ob-type'), text, { cps: 55 });
      // Swap the typed text for an editable field, preserving content.
      appear.innerHTML = `<textarea id="ob-appear-ta" class="onboard-textarea" rows="5">${esc(text)}</textarea>`;
      appear.querySelector('#ob-appear-ta').addEventListener('input', e => {
        data.appearanceDescription = e.target.value;
      });
    }

    card.querySelector('#ob-regen').addEventListener('click', () => {
      data.appearanceDescription = '';
      load();
    });
    card.querySelector('#ob-next').addEventListener('click', stepMemories);
    load();
  }

  // ── 4. A past — memories fly into the character ─────────────────────────────
  function stepMemories() {
    setStep(4);
    card.innerHTML = `
      <h2 class="onboard-h">Give ${esc(data.name)} a past</h2>
      <p class="onboard-subnote">Each memory becomes part of them. Add as many as you like.</p>
      <div class="onboard-mem-input">
        <input id="ob-mem-line" class="onboard-input" type="text"
               placeholder="A memory, a moment, a fact they carry… (Enter to add)" autocomplete="off" />
      </div>
      <ul class="onboard-mem-list" id="ob-mem-list"></ul>
      <label class="onboard-label" for="ob-notes">Backstory, scripts, or notes (optional)</label>
      <textarea id="ob-notes" class="onboard-textarea" rows="3"
                placeholder="Anything else worth knowing…">${esc(data.notes)}</textarea>
      <label class="onboard-label onboard-file-label" for="ob-files">＋ Attach docs (.txt, .md)</label>
      <input id="ob-files" class="onboard-file" type="file" accept=".txt,.md,text/*" multiple hidden />
      <div class="onboard-actions">
        <button id="ob-back" class="onboard-btn ghost">Back</button>
        <button id="ob-next" class="onboard-btn primary">${esc(data.name)} is ready</button>
      </div>`;

    const line = card.querySelector('#ob-mem-line');
    const list = card.querySelector('#ob-mem-list');
    const notes = card.querySelector('#ob-notes');
    const memArr = data.memories ? data.memories.split('\n').filter(Boolean) : [];

    const renderList = () => {
      list.innerHTML = memArr.map((m, i) =>
        `<li class="onboard-mem-item"><span>${esc(m)}</span><button data-i="${i}" class="onboard-mem-del" title="Remove">×</button></li>`
      ).join('');
      list.querySelectorAll('.onboard-mem-del').forEach(b =>
        b.addEventListener('click', () => { memArr.splice(+b.dataset.i, 1); data.memories = memArr.join('\n'); renderList(); }));
    };

    const addMemory = (text) => {
      const t = text.trim();
      if (!t) return;
      memArr.push(t);
      data.memories = memArr.join('\n');
      renderList();
      // The reward: the memory drifts into the character and is absorbed.
      flyInMemoryCard(overlay.querySelector('.onboard-stage'), charEl, t).catch(() => {});
      char?.greet?.();  // a small acknowledging motion
    };

    line.addEventListener('keydown', e => {
      if (e.key === 'Enter' && line.value.trim()) { addMemory(line.value); line.value = ''; }
    });
    notes.addEventListener('input', () => { data.notes = notes.value; });
    card.querySelector('#ob-file-trigger, .onboard-file-label')?.addEventListener('click', () => card.querySelector('#ob-files').click());
    card.querySelector('#ob-files').addEventListener('change', async e => {
      for (const f of Array.from(e.target.files || [])) {
        try {
          const text = await f.text();
          notes.value = (notes.value.trim() ? notes.value.trim() + '\n\n' : '') + `# ${f.name}\n${text}`;
        } catch (_) {}
      }
      data.notes = notes.value;
    });
    card.querySelector('#ob-back').addEventListener('click', stepAppearance);
    card.querySelector('#ob-next').addEventListener('click', stepBirth);
    renderList();
    setTimeout(() => line.focus(), 40);
  }

  // ── 5. Birth ────────────────────────────────────────────────────────────────
  function stepBirth() {
    setStep(5);
    radarWrap.classList.remove('visible');
    const cardEl = buildCharacterCard({
      name: data.name, archetype: data.archetype,
      ocean: data.ocean, appearance: data.appearanceDescription,
    });
    // Assemble into one wrapper, then mount in a single mutation so the
    // entrance animation fires exactly once.
    const wrap = document.createElement('div');
    wrap.className = 'onboard-birth';
    const h = document.createElement('h2');
    h.className = 'onboard-h onboard-birth-h';
    h.textContent = `Meet ${data.name}`;
    const actions = document.createElement('div');
    actions.className = 'onboard-actions';
    actions.innerHTML = `<button id="ob-launch" class="onboard-btn primary onboard-btn-lg">Bring them to life</button>`;
    wrap.append(h, cardEl, actions);
    card.replaceChildren(wrap);

    card.querySelector('#ob-launch').addEventListener('click', () => {
      confetti(charEl);
      char?.greet?.();
      setTimeout(finish, 650);   // let the burst + wave begin before handing off
    });
  }

  function finish() {
    const memLines = data.memories.split('\n').map(s => s.trim()).filter(Boolean);
    const noteLines = data.notes.split(/\n|(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);
    const characterConfig = {
      name: data.name,
      persona: [data.summary, data.notes].filter(s => s && s.trim()).join('\n\n'),
      ocean: { ...data.ocean },
      backstory: [...memLines, ...noteLines],
      facts: [],
      appearanceDescription: data.appearanceDescription,
      archetype: data.archetype,
    };
    radar?.destroy?.();
    cardObserver.disconnect();
    overlay.remove();
    onComplete?.(characterConfig);
  }

  stepName();
}

// ── helpers ────────────────────────────────────────────────────────────────
function normalizeOcean(o) {
  const out = {};
  for (const t of ['O', 'C', 'E', 'A', 'N']) {
    let v = Number(o[t]);
    if (!isFinite(v)) v = 0.5;
    if (v > 1) v = v / 100;
    out[t] = Math.max(0, Math.min(1, v));
  }
  return out;
}
function esc(s) {
  return String(s ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}
