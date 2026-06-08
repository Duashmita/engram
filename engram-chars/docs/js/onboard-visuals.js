/**
 * onboard-visuals.js
 *
 * Self-contained ES module providing the "rewarding" visual components for a
 * character-creation onboarding wizard in a dark-themed web app.
 *
 * Fully self-contained: injects its own CSS once (a single <style> appended to
 * document.head on first use). No external stylesheet, no external imports,
 * vanilla JS + Canvas 2D only.
 *
 * Palette:
 *   background #0d1117, surface #161c25, text #d6deeb,
 *   muted #7a8699, accent #6aa3ff
 *
 * All injected CSS classes are namespaced with `ov-`, EXCEPT `.character-card`
 * (kept verbatim because the wizard styles it too).
 *
 * ---------------------------------------------------------------------------
 * Exports & contracts:
 *
 * createFormingRadar(canvas) -> { setProgress(n), reveal(ocean), destroy() }
 *   Draws a 5-axis OCEAN radar (O,C,E,A,N), retina-aware.
 *   - Initial: faint pentagon grid + axis labels, no fill.
 *   - setProgress(n): n in 0..5. Grows a faint provisional "forming" blob;
 *     each answered question pushes axes out on a stable pseudo-random basis
 *     so the shape visibly builds. Animated. Cosmetic only.
 *   - reveal(ocean): ocean = {O,C,E,A,N} each 0..1. Animates morph to the TRUE
 *     shape over ~900ms with an accent filled polygon + glowing vertices.
 *     Returns Promise resolving on completion.
 *   - destroy(): cancels any RAF loop.
 *
 * typewriter(el, text, opts = {}) -> Promise<void>
 *   Types text into el.textContent char-by-char. opts.cps (default 45).
 *   Blinking caret via injected class during typing. opts.instant -> set
 *   immediately and resolve.
 *
 * revealArchetype(el, { archetype, summary }) -> Promise<void>
 *   Clears el; reveals an `.ov-arch-title` (fade+scale+glow) then an
 *   `.ov-arch-summary` paragraph filled via typewriter. Resolves when both done.
 *
 * flyInMemoryCard(containerEl, targetEl, text) -> Promise<void>
 *   Creates a small card (truncated text) absolutely positioned in containerEl,
 *   appears near the bottom/input area then drifts toward center of targetEl
 *   while shrinking + fading ("absorbed"). Removes element & resolves when done.
 *
 * buildCharacterCard({ name, archetype, ocean, appearance }) -> HTMLElement
 *   Returns a "collectible" `.character-card` element: name heading, accent
 *   archetype subtitle, inline 5-bar OCEAN readout, truncated appearance (~140).
 *   Caller inserts it into the DOM.
 *
 * confetti(originEl) -> void
 *   Brief tasteful particle burst from center of originEl via a temporary
 *   fixed overlay canvas that self-cleans after ~1.2s.
 * ---------------------------------------------------------------------------
 */

/* ===================== palette & constants ===================== */

const PALETTE = {
  bg: '#0d1117',
  surface: '#161c25',
  text: '#d6deeb',
  muted: '#7a8699',
  accent: '#6aa3ff',
};

const OCEAN_AXES = ['O', 'C', 'E', 'A', 'N'];

/* ===================== CSS injection (one-time) ===================== */

const STYLE_ID = 'ov-injected-styles';

function ensureStyles() {
  if (typeof document === 'undefined') return;
  if (document.getElementById(STYLE_ID)) return;

  const style = document.createElement('style');
  style.id = STYLE_ID;
  style.textContent = `
    @keyframes ov-caret-blink {
      0%, 49% { opacity: 1; }
      50%, 100% { opacity: 0; }
    }
    @keyframes ov-arch-in {
      from { opacity: 0; transform: translateY(8px) scale(0.94); }
      to   { opacity: 1; transform: translateY(0) scale(1); }
    }

    .ov-typing::after {
      content: '▌';
      margin-left: 1px;
      color: ${PALETTE.accent};
      animation: ov-caret-blink 1s step-end infinite;
    }

    .ov-arch-title {
      font-size: clamp(1.8rem, 4vw, 2.8rem);
      font-weight: 700;
      letter-spacing: 0.12em;
      color: ${PALETTE.accent};
      text-align: center;
      margin: 0 0 0.5em 0;
      opacity: 0;
      text-shadow: 0 0 18px rgba(106, 163, 255, 0.45),
                   0 0 4px rgba(106, 163, 255, 0.3);
      animation: ov-arch-in 700ms cubic-bezier(0.22, 1, 0.36, 1) forwards;
    }
    .ov-arch-summary {
      color: ${PALETTE.text};
      font-size: 1rem;
      line-height: 1.6;
      text-align: center;
      max-width: 46ch;
      margin: 0 auto;
      min-height: 1.6em;
    }

    .ov-mem-card {
      position: absolute;
      z-index: 50;
      max-width: 220px;
      padding: 8px 12px;
      border-radius: 10px;
      background: ${PALETTE.surface};
      color: ${PALETTE.text};
      border: 1px solid rgba(106, 163, 255, 0.25);
      box-shadow: 0 6px 22px rgba(0, 0, 0, 0.45);
      font-size: 0.85rem;
      line-height: 1.35;
      pointer-events: none;
      will-change: transform, opacity;
      white-space: normal;
      overflow: hidden;
    }

    /*
     * Card is a flex column so every section stacks with a guaranteed gap.
     * This is what prevents the OCEAN block and the appearance text from
     * overlapping: the gap is enforced by flow, not by per-child margins.
     * Near-white surface with a soft border so it reads cleanly on a light page.
     */
    .character-card {
      box-sizing: border-box;
      display: flex;
      flex-direction: column;
      gap: 12px;
      width: 280px;
      padding: 18px;
      border-radius: 14px;
      background: #ffffff;
      color: #1c2433;
      border: 1px solid rgba(106, 163, 255, 0.35);
      box-shadow: 0 1px 2px rgba(28, 36, 51, 0.06),
                  0 8px 24px rgba(28, 36, 51, 0.10);
      font-family: inherit;
    }
    .character-card .ov-cc-name {
      margin: 0;
      font-size: 1.25rem;
      font-weight: 700;
      line-height: 1.2;
      color: #1c2433;
    }
    .character-card .ov-cc-arch {
      margin: -6px 0 0 0; /* tuck close under the name, gap still applies below */
      font-size: 0.85rem;
      font-weight: 600;
      letter-spacing: 0.06em;
      color: ${PALETTE.accent};
      text-transform: uppercase;
    }
    .character-card .ov-cc-ocean {
      display: grid;
      grid-template-columns: 1ch 1fr auto;
      align-items: center;
      column-gap: 8px;
      row-gap: 8px;
      margin: 0;
    }
    .character-card .ov-cc-axis {
      font-size: 0.72rem;
      color: ${PALETTE.muted};
      font-weight: 600;
      line-height: 1;
      text-align: center;
    }
    .character-card .ov-cc-bar {
      height: 6px;
      border-radius: 3px;
      background: rgba(122, 134, 153, 0.22);
      overflow: hidden;
    }
    .character-card .ov-cc-bar > span {
      display: block;
      height: 100%;
      border-radius: 3px;
      background: ${PALETTE.accent};
    }
    .character-card .ov-cc-num {
      font-size: 0.72rem;
      color: #1c2433;
      font-variant-numeric: tabular-nums;
      min-width: 2.5ch;
      line-height: 1;
      text-align: right;
    }
    /*
     * Appearance sits clearly below the OCEAN block: the flex gap provides the
     * vertical separation and the top border plus padding-top draw a visible
     * divider so the scrollable text can never collide with the bars above it.
     */
    .character-card .ov-cc-appearance {
      margin: 0;
      padding: 12px 4px 0 0;
      border-top: 1px solid rgba(28, 36, 51, 0.10);
      font-size: 0.82rem;
      line-height: 1.5;
      color: ${PALETTE.muted};
      max-height: 7.5em; /* about 5 lines, then scroll. full text, no cutoff */
      overflow-y: auto;
    }

    .ov-confetti-overlay {
      position: fixed;
      inset: 0;
      pointer-events: none;
      z-index: 9999;
    }
  `;
  document.head.appendChild(style);
}

/* ===================== small helpers ===================== */

function clamp01(v) {
  v = Number(v);
  if (!isFinite(v)) return 0;
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function easeInOut(t) {
  // smooth ease-in-out
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function truncate(str, max) {
  str = String(str == null ? '' : str);
  if (str.length <= max) return str;
  return str.slice(0, Math.max(0, max - 1)).trimEnd() + '…';
}

// Stable pseudo-random value in [0,1] for a given index (deterministic).
function stableRand(i) {
  const x = Math.sin((i + 1) * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function now() {
  return (typeof performance !== 'undefined' && performance.now)
    ? performance.now()
    : Date.now();
}

/* ===================== 1. createFormingRadar ===================== */

export function createFormingRadar(canvas) {
  ensureStyles();

  if (!canvas || typeof canvas.getContext !== 'function') {
    // Defensive no-op object.
    return {
      setProgress() {},
      reveal() { return Promise.resolve(); },
      destroy() {},
    };
  }

  const ctx = canvas.getContext('2d');
  const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;

  let raf = 0;
  let destroyed = false;

  // current displayed per-axis radii (0..1) and target radii
  let current = [0, 0, 0, 0, 0];
  let target = [0, 0, 0, 0, 0];
  let revealed = false; // once true, draw accent filled polygon

  // logical (css) size
  let W = 0;
  let H = 0;

  function sizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    W = rect.width || canvas.width || 240;
    H = rect.height || canvas.height || 240;
    canvas.width = Math.max(1, Math.round(W * dpr));
    canvas.height = Math.max(1, Math.round(H * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function center() {
    return { cx: W / 2, cy: H / 2, R: Math.min(W, H) * 0.36 };
  }

  function axisPoint(cx, cy, R, i, r) {
    // start at top, clockwise
    const ang = -Math.PI / 2 + (i / 5) * Math.PI * 2;
    return {
      x: cx + Math.cos(ang) * R * r,
      y: cy + Math.sin(ang) * R * r,
      ang,
    };
  }

  function drawGrid(cx, cy, R) {
    // concentric pentagons
    ctx.lineWidth = 1;
    for (let ring = 1; ring <= 4; ring++) {
      const rr = ring / 4;
      ctx.beginPath();
      for (let i = 0; i < 5; i++) {
        const p = axisPoint(cx, cy, R, i, rr);
        if (i === 0) ctx.moveTo(p.x, p.y);
        else ctx.lineTo(p.x, p.y);
      }
      ctx.closePath();
      ctx.strokeStyle = `rgba(122, 134, 153, ${0.10 + ring * 0.015})`;
      ctx.stroke();
    }
    // spokes
    ctx.strokeStyle = 'rgba(122, 134, 153, 0.16)';
    for (let i = 0; i < 5; i++) {
      const p = axisPoint(cx, cy, R, i, 1);
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    // labels
    ctx.fillStyle = PALETTE.muted;
    ctx.font = '600 12px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < 5; i++) {
      const p = axisPoint(cx, cy, R, i, 1.16);
      ctx.fillText(OCEAN_AXES[i], p.x, p.y);
    }
  }

  function drawShape(cx, cy, R, radii) {
    const pts = radii.map((r, i) => axisPoint(cx, cy, R, i, r));

    // fill
    ctx.beginPath();
    pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.closePath();

    if (revealed) {
      ctx.fillStyle = 'rgba(106, 163, 255, 0.18)';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = PALETTE.accent;
      ctx.stroke();
      // glowing vertices
      pts.forEach((p) => {
        ctx.save();
        ctx.shadowColor = PALETTE.accent;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 3.2, 0, Math.PI * 2);
        ctx.fillStyle = PALETTE.accent;
        ctx.fill();
        ctx.restore();
      });
    } else {
      // faint provisional forming blob
      ctx.fillStyle = 'rgba(106, 163, 255, 0.07)';
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = 'rgba(106, 163, 255, 0.30)';
      ctx.stroke();
    }
  }

  function render() {
    sizeCanvas();
    ctx.clearRect(0, 0, W, H);
    const { cx, cy, R } = center();
    drawGrid(cx, cy, R);
    const hasShape = current.some((v) => v > 0.001);
    if (hasShape || revealed) drawShape(cx, cy, R, current);
  }

  // generic eased tween of current -> target over duration
  function animateTo(duration) {
    return new Promise((resolve) => {
      if (destroyed) return resolve();
      if (raf) cancelAnimationFrame(raf);
      const from = current.slice();
      const to = target.slice();
      const start = now();
      const step = () => {
        if (destroyed) return resolve();
        const t = Math.min(1, (now() - start) / duration);
        const e = easeInOut(t);
        for (let i = 0; i < 5; i++) {
          current[i] = from[i] + (to[i] - from[i]) * e;
        }
        render();
        if (t < 1) {
          raf = requestAnimationFrame(step);
        } else {
          current = to.slice();
          render();
          raf = 0;
          resolve();
        }
      };
      raf = requestAnimationFrame(step);
    });
  }

  // initial draw
  render();

  return {
    setProgress(n) {
      if (destroyed || revealed) return;
      n = Math.max(0, Math.min(5, Number(n) || 0));
      // Each answered question nudges axes outward on a stable per-axis basis.
      for (let i = 0; i < 5; i++) {
        // base growth proportional to progress, modulated by stable jitter
        const jitter = 0.45 + stableRand(i) * 0.5; // 0.45..0.95
        const base = (n / 5) * jitter;
        // add a little per-question texture so it "builds" unevenly
        const texture = stableRand(i * 7 + n) * 0.08;
        target[i] = clamp01(base * 0.7 + texture);
      }
      animateTo(450);
    },

    reveal(ocean) {
      if (destroyed) return Promise.resolve();
      ocean = ocean || {};
      revealed = true;
      target = OCEAN_AXES.map((k) => clamp01(ocean[k]));
      return animateTo(900);
    },

    destroy() {
      destroyed = true;
      if (raf) {
        cancelAnimationFrame(raf);
        raf = 0;
      }
    },
  };
}

/* ===================== 2. typewriter ===================== */

export function typewriter(el, text, opts = {}) {
  ensureStyles();

  if (!el) return Promise.resolve();
  text = String(text == null ? '' : text);

  if (opts.instant) {
    el.textContent = text;
    return Promise.resolve();
  }

  const cps = (opts.cps && opts.cps > 0) ? opts.cps : 45;
  const interval = 1000 / cps;

  return new Promise((resolve) => {
    el.textContent = '';
    el.classList.add('ov-typing');
    let i = 0;
    let last = now();
    let raf = 0;

    const step = () => {
      const t = now();
      const due = Math.floor((t - last) / interval);
      if (due > 0) {
        i = Math.min(text.length, i + due);
        last += due * interval;
        el.textContent = text.slice(0, i);
      }
      if (i < text.length) {
        raf = requestAnimationFrame(step);
      } else {
        el.classList.remove('ov-typing');
        if (raf) cancelAnimationFrame(raf);
        resolve();
      }
    };
    raf = requestAnimationFrame(step);
  });
}

/* ===================== 3. revealArchetype ===================== */

export function revealArchetype(el, { archetype, summary } = {}) {
  ensureStyles();

  if (!el) return Promise.resolve();

  el.textContent = '';

  const title = document.createElement('div');
  title.className = 'ov-arch-title';
  title.textContent = String(archetype == null ? '' : archetype);
  el.appendChild(title);

  const para = document.createElement('p');
  para.className = 'ov-arch-summary';
  el.appendChild(para);

  return new Promise((resolve) => {
    // wait for the title's fade+scale glow (~700ms) before typing summary
    const titleDelay = 750;
    const timer = setTimeout(() => {
      typewriter(para, String(summary == null ? '' : summary), { cps: 48 })
        .then(resolve);
    }, titleDelay);

    // safety: if setTimeout unavailable, resolve via fallback
    if (typeof timer === 'undefined') {
      typewriter(para, String(summary == null ? '' : summary), { cps: 48 })
        .then(resolve);
    }
  });
}

/* ===================== 4. flyInMemoryCard ===================== */

export function flyInMemoryCard(containerEl, targetEl, text) {
  ensureStyles();

  if (!containerEl || !targetEl) return Promise.resolve();

  const card = document.createElement('div');
  card.className = 'ov-mem-card';
  card.textContent = truncate(text, 90);

  // ensure container can host absolutely-positioned children
  const cs = (typeof getComputedStyle === 'function')
    ? getComputedStyle(containerEl)
    : null;
  if (cs && cs.position === 'static') {
    containerEl.style.position = 'relative';
  }

  containerEl.appendChild(card);

  return new Promise((resolve) => {
    // measure after insertion
    const contRect = containerEl.getBoundingClientRect();
    const cardRect = card.getBoundingClientRect();
    const tgtRect = targetEl.getBoundingClientRect();

    // start position: near bottom-center of the container (input area)
    const startX = contRect.width / 2 - cardRect.width / 2;
    const startY = contRect.height - cardRect.height - 16;

    // end position: center of target, relative to container
    const tgtCenterX = (tgtRect.left - contRect.left) + tgtRect.width / 2;
    const tgtCenterY = (tgtRect.top - contRect.top) + tgtRect.height / 2;
    const endX = tgtCenterX - cardRect.width / 2;
    const endY = tgtCenterY - cardRect.height / 2;

    card.style.left = startX + 'px';
    card.style.top = startY + 'px';
    card.style.transform = 'translate(0,0) scale(1)';
    card.style.opacity = '1';

    const duration = 850;
    const holdIn = 180; // brief appear before drifting
    const start = now();
    let raf = 0;

    const step = () => {
      const elapsed = now() - start;
      if (elapsed < holdIn) {
        raf = requestAnimationFrame(step);
        return;
      }
      const t = Math.min(1, (elapsed - holdIn) / duration);
      const e = easeInOut(t);
      const dx = (endX - startX) * e;
      const dy = (endY - startY) * e;
      const scale = 1 - 0.7 * e;
      card.style.transform = `translate(${dx}px, ${dy}px) scale(${scale})`;
      card.style.opacity = String(1 - e * 0.95);
      if (t < 1) {
        raf = requestAnimationFrame(step);
      } else {
        if (raf) cancelAnimationFrame(raf);
        if (card.parentNode) card.parentNode.removeChild(card);
        resolve();
      }
    };
    raf = requestAnimationFrame(step);
  });
}

/* ===================== 5. buildCharacterCard ===================== */

export function buildCharacterCard({ name, archetype, ocean, appearance } = {}) {
  ensureStyles();

  const card = document.createElement('div');
  card.className = 'character-card';

  const h = document.createElement('h3');
  h.className = 'ov-cc-name';
  h.textContent = truncate(name || 'Unnamed', 40);
  card.appendChild(h);

  const sub = document.createElement('div');
  sub.className = 'ov-cc-arch';
  sub.textContent = truncate(archetype || 'Unknown', 40);
  card.appendChild(sub);

  const grid = document.createElement('div');
  grid.className = 'ov-cc-ocean';
  const o = ocean || {};
  OCEAN_AXES.forEach((axis) => {
    const label = document.createElement('span');
    label.className = 'ov-cc-axis';
    label.textContent = axis;

    const bar = document.createElement('div');
    bar.className = 'ov-cc-bar';
    const fill = document.createElement('span');
    const v = clamp01(o[axis]);
    fill.style.width = (v * 100).toFixed(1) + '%';
    bar.appendChild(fill);

    const num = document.createElement('span');
    num.className = 'ov-cc-num';
    num.textContent = v.toFixed(2);

    grid.appendChild(label);
    grid.appendChild(bar);
    grid.appendChild(num);
  });
  card.appendChild(grid);

  // Full appearance description in a scrollable box (no truncation).
  const app = document.createElement('p');
  app.className = 'ov-cc-appearance';
  app.textContent = appearance || '';
  card.appendChild(app);

  return card;
}

/* ===================== 6. confetti ===================== */

export function confetti(originEl) {
  ensureStyles();

  if (!originEl || typeof document === 'undefined') return;

  const rect = originEl.getBoundingClientRect();
  const ox = rect.left + rect.width / 2;
  const oy = rect.top + rect.height / 2;

  const canvas = document.createElement('canvas');
  canvas.className = 'ov-confetti-overlay';
  const dpr = (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
  const w = (typeof window !== 'undefined' ? window.innerWidth : 800);
  const h = (typeof window !== 'undefined' ? window.innerHeight : 600);
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  document.body.appendChild(canvas);

  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const colors = [PALETTE.accent, PALETTE.accent, PALETTE.muted, '#9bbcff'];
  const N = 28;
  const particles = [];
  for (let i = 0; i < N; i++) {
    const ang = (i / N) * Math.PI * 2 + stableRand(i) * 0.6;
    const speed = 90 + stableRand(i * 3) * 140; // px/s
    particles.push({
      x: ox,
      y: oy,
      vx: Math.cos(ang) * speed,
      vy: Math.sin(ang) * speed - 40,
      r: 1.8 + stableRand(i * 5) * 2.2,
      color: colors[i % colors.length],
    });
  }

  const gravity = 220; // px/s^2
  const duration = 1100;
  const start = now();
  let last = start;
  let raf = 0;

  const step = () => {
    const t = now();
    const dt = Math.min(0.05, (t - last) / 1000);
    last = t;
    const elapsed = t - start;
    const life = Math.min(1, elapsed / duration);
    const alpha = 1 - life;

    ctx.clearRect(0, 0, w, h);
    for (const p of particles) {
      p.vy += gravity * dt;
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();
    }
    ctx.globalAlpha = 1;

    if (elapsed < duration) {
      raf = requestAnimationFrame(step);
    } else {
      if (raf) cancelAnimationFrame(raf);
      if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
    }
  };
  raf = requestAnimationFrame(step);
}
