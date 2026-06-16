// catalogue.js
// Light-themed start screen for the 3D NPC app.
// Lets the user preview a character in a 3D viewport, chat with it,
// or create a brand new one. Manages a saved-NPC catalogue in localStorage.
//
// Pure vanilla DOM for the layout. Face thumbnails are produced by the shared
// offscreen renderer in portrait.js (the only direct import here).
// All styles live in catalogue.css (the app links it). Classes use a cat- prefix.

import { getPortrait } from './portrait.js';

// Research links shown in the panel header.
const PAPER_URL = 'https://camps.aptaracorp.com/ACM_PMS/PMS/ACM/FDG26/102/03e19e98-4a72-11f1-b513-16ffd757ba29/OUT/fdg26-102.html';
const GITHUB_URL = 'https://github.com/Duashmita/engram';

const STORAGE_KEY = 'engram_catalogue';
const OCEAN_KEYS = ['O', 'C', 'E', 'A', 'N'];
const OCEAN_LABELS = {
  O: 'Openness',
  C: 'Conscientiousness',
  E: 'Extraversion',
  A: 'Agreeableness',
  N: 'Neuroticism'
};

// ---------------------------------------------------------------------------
// localStorage catalogue helpers
// ---------------------------------------------------------------------------

export function getCatalogue() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Keep only plausible entries.
    return parsed.filter((e) => e && typeof e === 'object' && e.id != null);
  } catch (err) {
    return [];
  }
}

function writeCatalogue(list) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch (err) {
    // Storage may be full or unavailable. Fail quietly.
  }
}

export function saveToCatalogue(entry) {
  if (!entry || typeof entry !== 'object' || entry.id == null) return entry;
  const saved = {
    id: entry.id,
    name: entry.name || 'Unnamed',
    archetype: entry.archetype || 'Custom character',
    ocean: normalizeOcean(entry.ocean),
    persona: entry.persona || '',
    // backstory/facts must stay arrays: the backend's /start expects lists.
    backstory: Array.isArray(entry.backstory)
      ? entry.backstory
      : (entry.backstory ? [String(entry.backstory)] : []),
    facts: Array.isArray(entry.facts) ? entry.facts : [],
    appearanceDescription: entry.appearanceDescription || '',
    source: 'custom',
    assetPath: entry.assetPath,
    glbUrl: entry.glbUrl
  };
  const list = getCatalogue();
  const idx = list.findIndex((e) => e.id === saved.id);
  if (idx >= 0) {
    list[idx] = saved;
  } else {
    list.push(saved);
  }
  writeCatalogue(list);
  return saved;
}

export function removeFromCatalogue(id) {
  if (id == null) return;
  const list = getCatalogue().filter((e) => e.id !== id);
  writeCatalogue(list);
}

// ---------------------------------------------------------------------------
// Small utilities
// ---------------------------------------------------------------------------

function normalizeOcean(ocean) {
  const out = {};
  const src = ocean && typeof ocean === 'object' ? ocean : {};
  for (const k of OCEAN_KEYS) {
    let v = Number(src[k]);
    if (!Number.isFinite(v)) v = 0.5;
    // Clamp to a 0..1 range. Accept 0..100 inputs too.
    if (v > 1) v = v / 100;
    if (v < 0) v = 0;
    if (v > 1) v = 1;
    out[k] = v;
  }
  return out;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}

// First word of a name, used for short button labels and monogram initials.
function firstName(name) {
  const n = (name || '').trim();
  if (!n) return '';
  return n.split(/\s+/)[0];
}

// Stable HSL color derived from the name, for the monogram fallback circle.
function colorFromName(name) {
  const s = name || '?';
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) % 360;
  }
  return `hsl(${h}, 52%, 58%)`;
}

const OCEAN_TIP =
  'OCEAN is the Big Five personality model: Openness, Conscientiousness, ' +
  'Extraversion, Agreeableness, Neuroticism, each from 0 to 1. Personality ' +
  'shapes what this character remembers, retrieves, and forgets.';

// Build an info "i" dot per the shared markup contract. CSS for .info-dot /
// .tip-bubble is provided by style.css (loaded on the same page).
function buildInfoDot(tip) {
  const dot = document.createElement('span');
  dot.className = 'info-dot';
  dot.setAttribute('data-tip', tip);
  dot.setAttribute('tabindex', '0');
  dot.setAttribute('role', 'img');
  dot.setAttribute('aria-label', 'More info');
  dot.textContent = 'i';
  return dot;
}

function buildOceanReadout(ocean) {
  // Outer block: a small "OCEAN" label + info dot, then the five bars.
  const block = el('div', 'cat-ocean-block');

  const labelRow = el('div', 'cat-ocean-label');
  labelRow.appendChild(el('span', 'cat-ocean-label-text', 'OCEAN'));
  const dot = buildInfoDot(OCEAN_TIP);
  // The dot is interactive; don't let a click bubble up to row selection.
  dot.addEventListener('click', function (ev) { ev.stopPropagation(); });
  labelRow.appendChild(dot);
  block.appendChild(labelRow);

  const wrap = el('div', 'cat-ocean');
  wrap.setAttribute('aria-hidden', 'true');
  const norm = normalizeOcean(ocean);
  for (const k of OCEAN_KEYS) {
    const v = norm[k];
    const cell = el('span', 'cat-ocean-cell');
    cell.title = OCEAN_LABELS[k] + ': ' + Math.round(v * 100);
    const fill = el('span', 'cat-ocean-fill');
    fill.style.height = Math.max(8, Math.round(v * 100)) + '%';
    cell.appendChild(fill);
    const tag = el('span', 'cat-ocean-tag', k);
    cell.appendChild(tag);
    wrap.appendChild(cell);
  }
  block.appendChild(wrap);
  return block;
}

// Build the left-side portrait: a colored monogram circle shown immediately,
// then (async) a rendered face thumbnail that fades in if one is available.
function buildPortrait(entry) {
  const holder = el('div', 'cat-row-portrait');
  holder.setAttribute('aria-hidden', 'true');

  const fn = firstName(entry.name);
  const initial = (fn ? fn.charAt(0) : (entry.name || '?').charAt(0) || '?').toUpperCase();
  const mono = el('span', 'cat-portrait-mono', initial);
  mono.style.background = colorFromName(entry.name);
  holder.appendChild(mono);

  // Render a face in the background; swap it in with a soft fade if it works.
  Promise.resolve()
    .then(function () { return getPortrait(entry); })
    .then(function (dataUrl) {
      if (!dataUrl || !holder.isConnected) return;
      const img = el('img', 'cat-portrait-img');
      img.alt = '';
      img.decoding = 'async';
      img.addEventListener('load', function () {
        img.classList.add('cat-portrait-img-in');
      });
      img.src = dataUrl;
      holder.appendChild(img);
    })
    .catch(function () {
      // Keep the monogram on any failure.
    });

  return holder;
}

// ---------------------------------------------------------------------------
// Start screen
// ---------------------------------------------------------------------------

export function showStartScreen(options) {
  const opts = options || {};
  const mountEl = opts.mountEl;
  const onPreview = typeof opts.onPreview === 'function' ? opts.onPreview : function () {};
  const onPlay = typeof opts.onPlay === 'function' ? opts.onPlay : function () {};
  const onEdit = typeof opts.onEdit === 'function' ? opts.onEdit : function () {};
  const onCreateNew = typeof opts.onCreateNew === 'function' ? opts.onCreateNew : function () {};
  const presets = Array.isArray(opts.presets) ? opts.presets : [];

  if (!mountEl || !mountEl.appendChild) {
    // Nothing to mount into. Return a no-op handle.
    return { destroy: function () {} };
  }

  // State.
  let entries = buildEntries();
  let activeId = null;
  let rowEls = new Map();

  function buildEntries() {
    const customs = getCatalogue();
    const presetList = presets.map((p) => Object.assign({ source: 'preset' }, p));
    const customList = customs.map((c) => Object.assign({}, c, { source: 'custom' }));
    return presetList.concat(customList);
  }

  // --- Overlay scaffold -----------------------------------------------------
  const overlay = el('div', 'cat-overlay');

  const grid = el('div', 'cat-grid');
  overlay.appendChild(grid);

  // Left: viewport.
  const viewport = el('div', 'cat-viewport');
  const canvas = document.createElement('canvas');
  canvas.id = 'cat-canvas';
  canvas.className = 'cat-canvas';
  viewport.appendChild(canvas);

  const playBar = el('div', 'cat-playbar');
  const playBtn = el('button', 'cat-play-btn');
  playBtn.type = 'button';
  playBtn.disabled = true;
  playBtn.textContent = 'Chat';
  playBar.appendChild(playBtn);
  viewport.appendChild(playBar);

  grid.appendChild(viewport);

  // Right: panel.
  const panel = el('div', 'cat-panel');

  const header = el('div', 'cat-header');
  const headerTop = el('div', 'cat-header-top');
  headerTop.appendChild(el('div', 'cat-brand', 'Engram'));
  const links = el('nav', 'cat-links');
  links.setAttribute('aria-label', 'Project links');
  const paperLink = el('a', 'cat-link cat-link-cta', 'Paper');
  paperLink.href = PAPER_URL;
  paperLink.target = '_blank';
  paperLink.rel = 'noopener';
  const ghLink = el('a', 'cat-link', 'GitHub');
  ghLink.href = GITHUB_URL;
  ghLink.target = '_blank';
  ghLink.rel = 'noopener';
  links.appendChild(paperLink);
  links.appendChild(ghLink);
  headerTop.appendChild(links);
  header.appendChild(headerTop);
  header.appendChild(el('h1', 'cat-title', 'Choose a character'));
  header.appendChild(el('p', 'cat-subtitle', 'Pick someone to talk to, or build your own.'));
  panel.appendChild(header);

  const createCard = el('button', 'cat-create');
  createCard.type = 'button';
  const createIcon = el('span', 'cat-create-icon', '+');
  createIcon.setAttribute('aria-hidden', 'true');
  createCard.appendChild(createIcon);
  const createText = el('span', 'cat-create-text');
  createText.appendChild(el('span', 'cat-create-title', 'Make your own Character'));
  createText.appendChild(el('span', 'cat-create-sub', 'Design a persona and personality from scratch'));
  createCard.appendChild(createText);
  createCard.addEventListener('click', function () {
    onCreateNew();
  });
  panel.appendChild(createCard);

  const list = el('div', 'cat-list');
  list.setAttribute('role', 'listbox');
  list.setAttribute('aria-label', 'Available characters');
  panel.appendChild(list);

  grid.appendChild(panel);

  mountEl.appendChild(overlay);

  // --- Selection ------------------------------------------------------------
  function selectEntry(entry) {
    if (!entry) return;
    activeId = entry.id;

    // Update active markers.
    for (const [id, node] of rowEls.entries()) {
      const isActive = id === activeId;
      node.classList.toggle('cat-row-active', isActive);
      node.setAttribute('aria-selected', isActive ? 'true' : 'false');
    }

    // Reveal and wire the play button.
    playBtn.disabled = false;
    playBtn.textContent = 'Chat with ' + (entry.name || 'character');
    playBar.classList.add('cat-playbar-on');

    try {
      onPreview(entry, canvas);
    } catch (err) {
      // App-side preview failure should not break the start screen.
    }
  }

  playBtn.addEventListener('click', function () {
    const entry = entries.find((e) => e.id === activeId);
    if (entry) onPlay(entry);
  });

  // --- List rendering -------------------------------------------------------
  function renderList() {
    list.innerHTML = '';
    rowEls = new Map();

    if (!entries.length) {
      list.appendChild(el('div', 'cat-empty', 'No characters yet. Create one to get started.'));
      return;
    }

    entries.forEach(function (entry) {
      const row = el('div', 'cat-row');
      row.setAttribute('role', 'option');
      row.setAttribute('tabindex', '0');
      row.setAttribute('aria-selected', 'false');

      // Portrait: monogram fallback first, then swap in a rendered face.
      row.appendChild(buildPortrait(entry));

      const info = el('div', 'cat-row-info');
      info.appendChild(el('div', 'cat-row-name', entry.name || 'Unnamed'));
      info.appendChild(el('div', 'cat-row-archetype', entry.archetype || ''));
      info.appendChild(buildOceanReadout(entry.ocean));
      row.appendChild(info);

      // Hover/focus actions: a primary Chat button, plus Edit for customs.
      const actions = el('div', 'cat-row-actions');

      const fn = firstName(entry.name);
      const chatBtn = el('button', 'cat-row-chat');
      chatBtn.type = 'button';
      chatBtn.textContent = (fn && fn.length <= 12) ? ('Chat with ' + fn) : 'Chat';
      chatBtn.setAttribute('aria-label', 'Chat with ' + (entry.name || 'character'));
      chatBtn.addEventListener('click', function (ev) {
        ev.stopPropagation();
        onPlay(entry);
      });
      actions.appendChild(chatBtn);

      if (entry.source === 'custom') {
        const editBtn = el('button', 'cat-row-edit');
        editBtn.type = 'button';
        editBtn.textContent = 'Edit';
        editBtn.setAttribute('aria-label', 'Edit ' + (entry.name || 'character'));
        editBtn.addEventListener('click', function (ev) {
          ev.stopPropagation();
          onEdit(entry);
        });
        actions.appendChild(editBtn);
      }

      row.appendChild(actions);

      if (entry.source === 'custom') {
        const removeBtn = el('button', 'cat-remove');
        removeBtn.type = 'button';
        removeBtn.textContent = 'x';
        removeBtn.setAttribute('aria-label', 'Remove ' + (entry.name || 'character'));
        removeBtn.addEventListener('click', function (ev) {
          ev.stopPropagation();
          removeFromCatalogue(entry.id);
          refresh();
        });
        row.appendChild(removeBtn);
      }

      row.addEventListener('click', function () {
        selectEntry(entry);
      });
      row.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          selectEntry(entry);
        }
      });

      rowEls.set(entry.id, row);
      list.appendChild(row);
    });
  }

  // Rebuild entries from sources, render, and keep a sensible selection.
  function refresh() {
    entries = buildEntries();
    renderList();

    if (!entries.length) {
      activeId = null;
      playBtn.disabled = true;
      playBtn.textContent = 'Chat';
      playBar.classList.remove('cat-playbar-on');
      return;
    }

    const stillThere = entries.some((e) => e.id === activeId);
    if (!stillThere) {
      selectEntry(entries[0]);
    } else {
      // Re-apply active markers after re-render.
      const current = entries.find((e) => e.id === activeId);
      selectEntry(current);
    }
  }

  // Initial paint and auto-select the first entry.
  refresh();

  // --- Handle ---------------------------------------------------------------
  return {
    destroy: function () {
      if (overlay && overlay.parentNode) {
        overlay.parentNode.removeChild(overlay);
      }
    }
  };
}
