// portrait.js
// Shared offscreen Three.js renderer that turns a character GLB into a small
// head-and-shoulders thumbnail (a "bust" shot) and returns a data URL.
//
// Design notes:
// - ONE hidden offscreen canvas + ONE WebGLRenderer, created lazily and reused
//   for every portrait. Browsers cap the number of live WebGL contexts, so we
//   must never create a renderer per call.
// - Calls are processed SEQUENTIALLY via a promise chain (one model in flight
//   at a time) to avoid GPU thrash.
// - Results are cached in memory and in localStorage so revisits are instant.
// - Any failure (no url, WebGL unavailable, load error) resolves null and never
//   throws, so callers can always render a monogram fallback safely.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { BACKEND_URL } from '../config.js';

const SIZE = 256;
const STORAGE_KEY = 'engram_portraits_v1';
// Cap the persisted portrait blob so this single key can't grow unbounded
// toward the localStorage quota. Oldest entries are evicted first.
const MAX_STORE = 40;

// In-memory cache keyed by resolved GLB url -> data URL string.
const memCache = new Map();
// Tracks in-flight promises so the same url isn't rendered twice concurrently.
const inflight = new Map();

// Lazily initialized rendering context (renderer, scene, camera, loader, lights).
let ctx = null;
// Promise chain that serializes work: each portrait waits for the previous one.
let queue = Promise.resolve();

// ---------------------------------------------------------------------------
// localStorage cache (best-effort)
// ---------------------------------------------------------------------------

function loadStore() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === 'object' ? parsed : {};
  } catch (err) {
    return {};
  }
}

function readStore(url) {
  const store = loadStore();
  const val = store[url];
  return typeof val === 'string' ? val : null;
}

function writeStore(url, dataUrl) {
  try {
    const store = loadStore();
    store[url] = dataUrl;
    // Evict oldest entries (insertion order is preserved for string keys) so
    // the blob stays bounded.
    const keys = Object.keys(store);
    if (keys.length > MAX_STORE) {
      for (const k of keys.slice(0, keys.length - MAX_STORE)) delete store[k];
    }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  } catch (err) {
    // Storage may be full or unavailable. Ignore.
  }
}

// ---------------------------------------------------------------------------
// Renderer setup (lazy, once)
// ---------------------------------------------------------------------------

function getContext() {
  if (ctx) return ctx;
  if (ctx === false) return null; // previously failed; don't retry forever.

  try {
    const canvas = document.createElement('canvas');
    canvas.width = SIZE;
    canvas.height = SIZE;
    // Keep the canvas out of layout/view entirely.
    canvas.style.position = 'absolute';
    canvas.style.left = '-9999px';
    canvas.style.top = '-9999px';
    canvas.style.width = SIZE + 'px';
    canvas.style.height = SIZE + 'px';
    canvas.setAttribute('aria-hidden', 'true');

    const renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: true,
      preserveDrawingBuffer: true // needed so toDataURL captures the frame
    });
    renderer.setPixelRatio(1);
    renderer.setSize(SIZE, SIZE, false);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    const scene = new THREE.Scene();
    scene.background = null;

    // Square camera; framed each render onto the head + shoulders.
    const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 100);

    // Lighting mirrors character.js (key / fill / back + ambient).
    scene.add(new THREE.AmbientLight(0xd0e0ff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.7);
    key.position.set(1.5, 3.5, 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0x8090c8, 0.5);
    fill.position.set(-2.5, 1.5, 1.5);
    scene.add(fill);
    const back = new THREE.DirectionalLight(0x304060, 0.35);
    back.position.set(0, 2.5, -3);
    scene.add(back);

    const loader = new GLTFLoader();

    ctx = { renderer, scene, camera, loader };
    return ctx;
  } catch (err) {
    // WebGL unavailable. Remember the failure so we don't keep trying.
    ctx = false;
    return null;
  }
}

// ---------------------------------------------------------------------------
// Model helpers
// ---------------------------------------------------------------------------

function loadGLTF(loader, url) {
  return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));
}

// Normalize like character.js loadBase: scale to ~1.85m tall, center on X/Z,
// sit the feet on y=0. This gives us a predictable head position to frame.
function normalize(model) {
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const scale = 1.85 / Math.max(size.y, 0.01);
  model.scale.setScalar(scale);
  model.position.set(-center.x, -box.min.y, -center.z);
  // Height in world units after scaling (~1.85).
  return Math.max(size.y, 0.01) * scale;
}

// Frame the camera on the upper body / head so the FACE is visible.
function frameBust(camera, height) {
  // After normalization the figure stands on y=0 with total height ~`height`.
  // The head sits near the top; aim a touch below the crown so the face and a
  // little shoulder are in frame.
  const headY = height * 0.88;
  camera.position.set(0, headY, height * 0.95);
  camera.lookAt(0, headY, 0);
  camera.updateProjectionMatrix();
}

function disposeModel(scene, model) {
  if (!model) return;
  scene.remove(model);
  model.traverse((n) => {
    if (n.isMesh) {
      if (n.geometry && n.geometry.dispose) n.geometry.dispose();
      const mat = n.material;
      if (Array.isArray(mat)) {
        mat.forEach((m) => disposeMaterial(m));
      } else if (mat) {
        disposeMaterial(mat);
      }
    }
  });
}

function disposeMaterial(mat) {
  for (const key in mat) {
    const val = mat[key];
    if (val && val.isTexture && val.dispose) val.dispose();
  }
  if (mat.dispose) mat.dispose();
}

function captureDataURL(renderer) {
  try {
    return renderer.domElement.toDataURL('image/webp', 0.85);
  } catch (err) {
    try {
      return renderer.domElement.toDataURL('image/png');
    } catch (err2) {
      return null;
    }
  }
}

// ---------------------------------------------------------------------------
// URL resolution
// ---------------------------------------------------------------------------

function resolveUrl(entry) {
  if (!entry || typeof entry !== 'object') return null;
  if (entry.source === 'preset' && entry.assetPath) {
    return `${entry.assetPath}/base.glb`;
  }
  if (entry.assetPath && entry.source !== 'custom') {
    return `${entry.assetPath}/base.glb`;
  }
  if (entry.glbUrl) {
    return `${BACKEND_URL}/proxy_glb?url=${encodeURIComponent(entry.glbUrl)}`;
  }
  // Presets may carry assetPath even when source isn't set; fall back to it.
  if (entry.assetPath) {
    return `${entry.assetPath}/base.glb`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Core render (one model at a time)
// ---------------------------------------------------------------------------

async function renderPortrait(url) {
  const context = getContext();
  if (!context) return null;
  const { renderer, scene, camera, loader } = context;

  let model = null;
  try {
    const gltf = await loadGLTF(loader, url);
    model = gltf.scene;
    scene.add(model);
    const height = normalize(model);
    frameBust(camera, height);
    renderer.render(scene, camera);
    const dataUrl = captureDataURL(renderer);
    return dataUrl;
  } catch (err) {
    return null;
  } finally {
    disposeModel(scene, model);
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * getPortrait(entry) -> Promise<string|null>
 * Resolves a data-URL face thumbnail for the character, or null on failure.
 * Results are cached in memory and localStorage. Work is serialized so only
 * one model is rendered at a time. Never throws.
 */
export function getPortrait(entry) {
  let url;
  try {
    url = resolveUrl(entry);
  } catch (err) {
    url = null;
  }
  if (!url) return Promise.resolve(null);

  // Memory cache.
  if (memCache.has(url)) return Promise.resolve(memCache.get(url));

  // localStorage cache.
  const stored = readStore(url);
  if (stored) {
    memCache.set(url, stored);
    return Promise.resolve(stored);
  }

  // Already rendering this exact url? Share the promise.
  if (inflight.has(url)) return inflight.get(url);

  // Enqueue behind any in-progress portrait so only one model is in flight.
  const task = queue.then(() => renderPortrait(url)).then(
    (dataUrl) => {
      if (dataUrl) {
        memCache.set(url, dataUrl);
        writeStore(url, dataUrl);
      } else if (ctx === false) {
        // WebGL is unavailable: a deterministic failure for every url. Cache
        // null so we stop trying. A transient load error (ctx still valid) is
        // NOT cached, so a later re-render can retry once the network recovers.
        memCache.set(url, null);
      }
      inflight.delete(url);
      return dataUrl;
    },
    () => {
      inflight.delete(url);
      return null;
    }
  );

  // Keep the queue going even if this task rejected (it shouldn't).
  queue = task.catch(() => {});
  inflight.set(url, task);
  return task;
}
