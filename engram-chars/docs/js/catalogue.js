// catalogue.js
// Light-themed start screen for the 3D NPC app.
// Lets the user preview a character in a 3D viewport, chat with it,
// or create a brand new one. Manages a saved-NPC catalogue in localStorage.
//
// Pure vanilla JS + DOM. No three.js; the only import is a lazy one of the
// sibling waitlist.js module when the waitlist button is clicked.
// All styles live in catalogue.css (the app links it). Classes use a cat- prefix.

// Research links shown in the panel header.
// TODO: set real URL for the paper and the repository.
const PAPER_URL = '#';
const GITHUB_URL = 'https://github.com/';

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

function buildOceanReadout(ocean) {
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
  return wrap;
}

// ---------------------------------------------------------------------------
// Start screen
// ---------------------------------------------------------------------------

export function showStartScreen(options) {
  const opts = options || {};
  const mountEl = opts.mountEl;
  const onPreview = typeof opts.onPreview === 'function' ? opts.onPreview : function () {};
  const onPlay = typeof opts.onPlay === 'function' ? opts.onPlay : function () {};
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
  const paperLink = el('a', 'cat-link', 'Paper');
  paperLink.href = PAPER_URL;
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
  createText.appendChild(el('span', 'cat-create-title', 'Create new character'));
  createText.appendChild(el('span', 'cat-create-sub', 'Start from a blank persona'));
  createCard.appendChild(createText);
  createCard.addEventListener('click', function () {
    onCreateNew();
  });
  panel.appendChild(createCard);

  // Waitlist entry point: lazy-load the dialog wiring so the start screen
  // stays dependency-free until the user actually clicks.
  const waitlistBtn = el('button', 'cat-waitlist', 'Want these characters in your game? Join the waitlist');
  waitlistBtn.type = 'button';
  waitlistBtn.addEventListener('click', function () {
    import('./waitlist.js')
      .then(function (m) { m.openWaitlist(); })
      .catch(function () {
        window.location.href = 'mailto:asdua@ucsc.edu?subject=' + encodeURIComponent('Engram waitlist');
      });
  });
  panel.appendChild(waitlistBtn);

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

      const info = el('div', 'cat-row-info');
      info.appendChild(el('div', 'cat-row-name', entry.name || 'Unnamed'));
      info.appendChild(el('div', 'cat-row-archetype', entry.archetype || ''));
      info.appendChild(buildOceanReadout(entry.ocean));
      row.appendChild(info);

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
