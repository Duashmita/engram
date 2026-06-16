// tooltip.js — lightweight, never-clipping hover/focus tooltips.
//
// Any element carrying a `data-tip="…"` attribute shows a floating bubble on
// hover or keyboard focus. The bubble is a single element appended to <body>
// and positioned with `position: fixed`, so it is never clipped by a panel's
// overflow (unlike a CSS ::after tooltip). Wire it once at startup via
// initTooltips(); it works for elements added later too (event delegation).

let _bubble = null;
let _wired = false;

function ensureBubble() {
  if (_bubble) return _bubble;
  _bubble = document.createElement('div');
  _bubble.className = 'tip-bubble';
  _bubble.setAttribute('role', 'tooltip');
  document.body.appendChild(_bubble);
  return _bubble;
}

function show(target) {
  const tip = target.getAttribute('data-tip');
  if (!tip) return;
  const b = ensureBubble();
  b.textContent = tip;
  // Make it measurable before positioning.
  b.style.left = '-9999px';
  b.style.top = '0px';
  b.classList.add('on');

  const r = target.getBoundingClientRect();
  const bw = b.offsetWidth;
  const bh = b.offsetHeight;
  const pad = 8;

  let left = r.left + r.width / 2 - bw / 2;
  left = Math.max(pad, Math.min(left, window.innerWidth - bw - pad));

  let top = r.top - bh - pad;          // prefer above
  if (top < pad) top = r.bottom + pad; // flip below if no room

  b.style.left = left + 'px';
  b.style.top = top + 'px';
}

function hide() {
  if (_bubble) _bubble.classList.remove('on');
}

export function initTooltips() {
  if (_wired) return;
  _wired = true;
  const closest = (el) => (el && el.closest ? el.closest('[data-tip]') : null);

  document.addEventListener('pointerover', (e) => {
    const t = closest(e.target);
    if (t) show(t);
  });
  document.addEventListener('pointerout', (e) => {
    if (closest(e.target)) hide();
  });
  // Keyboard accessibility: show on focus, hide on blur.
  document.addEventListener('focusin', (e) => {
    const t = closest(e.target);
    if (t) show(t);
  });
  document.addEventListener('focusout', hide);
  // Any scroll moves the anchor out from under the bubble — just hide it.
  window.addEventListener('scroll', hide, true);
}
