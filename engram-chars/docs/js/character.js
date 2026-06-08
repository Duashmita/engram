/**
 * character.js — Three.js NPC character loader and animator.
 *
 * Loads a Meshy-generated rigged GLB (base.glb) and blends animation clips
 * from sibling GLBs (idle.glb, talking.glb, etc.) using THREE.AnimationMixer.
 *
 * Design decisions:
 * - Progressive: shows the character immediately after base.glb loads, then
 *   loads animation clips in background via Promise.all().
 * - Track normalization: strips leading path components from animation track
 *   names so clips from a different GLTF hierarchy still bind correctly.
 * - Graceful degradation: placeholder capsule when no Meshy GLB exists.
 * - Personality idle: reads the manifest's `idle_anim` field to know which
 *   clip to use as the default personality-appropriate idle.
 */

import * as THREE from 'three';
import { GLTFLoader }      from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls }   from 'three/addons/controls/OrbitControls.js';
import * as SkeletonUtils  from 'three/addons/utils/SkeletonUtils.js';

const loader = new GLTFLoader();

function loadGLTF(url) {
  return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));
}

/** Build a set of bone names in the loaded model. */
function getBoneNames(object) {
  const names = new Set();
  object.traverse(n => { if (n.isBone) names.add(n.name); });
  return names;
}

/**
 * Strip position and scale tracks from non-root bones.
 *
 * Meshy animation clips include .position and .scale tracks on every bone.
 * For a standard humanoid rig, only the root bone (Hips) should translate —
 * all other bones express pose purely through rotation (.quaternion).
 * Leaving position tracks on arm/leg bones makes them fly away from the body.
 */
function prepareClip(clip) {
  const c = clip.clone();
  c.tracks = c.tracks.filter(track => {
    if (track.name.endsWith('.quaternion')) return true;   // always keep rotations
    if (track.name === 'Hips.position')     return true;   // keep root translation
    if (track.name === 'Hips.scale')        return false;  // drop root scale too
    if (track.name.endsWith('.position'))   return false;  // drop non-root positions
    if (track.name.endsWith('.scale'))      return false;  // drop all scale tracks
    return true;
  });
  return c;
}

export async function createCharacter(canvas, assetBasePath) {
  // ── Scene ──────────────────────────────────────────────────────────────────
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  scene.background = null;

  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
  camera.position.set(0, 1.5, 3.2);
  camera.lookAt(0, 1.0, 0);

  // Lighting
  scene.add(new THREE.AmbientLight(0xd0e0ff, 0.5));
  const key = new THREE.DirectionalLight(0xffffff, 1.6);
  key.position.set(2, 5, 3); key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x8090c8, 0.45);
  fill.position.set(-3, 2, -2);
  scene.add(fill);
  const back = new THREE.DirectionalLight(0x304060, 0.3);
  back.position.set(0, 3, -4);
  scene.add(back);

  // Ground
  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(1.8, 64),
    new THREE.MeshStandardMaterial({ color: 0x12182a, roughness: 1, metalness: 0 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  // Controls
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0, 1.0, 0);
  controls.minDistance = 1.5; controls.maxDistance = 7;
  controls.maxPolarAngle = Math.PI / 1.7;
  controls.enablePan = false;
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  // ── State ──────────────────────────────────────────────────────────────────
  let mixer         = null;
  let model         = null;
  let clips         = {};
  let currentAction = null;
  let personalityIdle = 'idle';
  let boneNames     = new Set();

  // ── Idle / breathing + look-at state ────────────────────────────────────────
  let breathingOn   = true;          // "alive" idle, ON by default after load
  let lookAtOn      = false;         // follow pointer on Y axis
  let elapsed       = 0;             // running time accumulator (seconds)
  let modelHeight   = 1.85;          // used for bob amplitude; updated on load
  // Base transform captured once the model is placed, so breathing/look-at are
  // applied RELATIVE to it (and never accumulate drift).
  let baseSet       = false;
  const basePos     = new THREE.Vector3();
  const baseScale   = new THREE.Vector3(1, 1, 1);
  let baseRotY      = 0;
  let targetYaw     = 0;             // desired look-at yaw (rad)
  // Procedural greet (nod + hop) state: when active, drives root bob/rot.
  let greetT        = 0;             // remaining time (s); 0 = inactive
  let greetDur      = 0;
  let materializing = false;         // true while materialize() ramps scale/opacity

  /** Capture the model's resting transform as the breathing baseline. */
  function captureBase() {
    if (!model) return;
    basePos.copy(model.position);
    baseScale.copy(model.scale);
    baseRotY = model.rotation.y;
    baseSet = true;
  }

  // ── Placeholder capsule ────────────────────────────────────────────────────
  function buildPlaceholder() {
    const group = new THREE.Group();
    const bodyGeo = new THREE.CapsuleGeometry(0.28, 1.1, 8, 16);
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2a3a5c, roughness: 0.7 });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 0.95; body.castShadow = true;
    group.add(body);

    const eyeGeo = new THREE.SphereGeometry(0.055, 8, 8);
    const eyeMat = new THREE.MeshStandardMaterial({ color: 0x88aaff, emissive: 0x3355bb, emissiveIntensity: 0.8 });
    for (const x of [-0.12, 0.12]) {
      const eye = new THREE.Mesh(eyeGeo, eyeMat);
      eye.position.set(x, 1.58, 0.26);
      group.add(eye);
    }
    return group;
  }

  // ── Load model + animations ────────────────────────────────────────────────
  // Meshy animation GLBs are self-contained: each (idle.glb, talking.glb, …) is
  // the FULL rigged character with one animation baked in, all sharing the same
  // rig. So we load the idle GLB AS the model — mesh and idle animation come
  // from the identical file, eliminating any cross-file binding mismatch — then
  // pull the other clips from sibling GLBs (they bind by bone name to the same
  // skeleton).
  async function loadBase() {
    if (!assetBasePath) { model = buildPlaceholder(); scene.add(model); modelHeight = 1.85; captureBase(); return; }

    // Manifest tells us which animations exist + the personality idle name.
    let animNames = [];
    let idleName = 'idle';
    try {
      const manifest = await fetch(`${assetBasePath}/manifest.json`).then(r => r.json());
      animNames = Object.keys(manifest.animations || {});
      if (manifest.idle_anim && animNames.includes(manifest.idle_anim)) {
        idleName = manifest.idle_anim;
      }
    } catch (_) {
      model = buildPlaceholder(); scene.add(model); modelHeight = 1.85; captureBase(); return;
    }
    personalityIdle = idleName;

    // The GLB we load as the actual displayed model — prefer the idle one.
    const primaryName = animNames.includes(idleName) ? idleName
                      : (animNames[0] || null);
    if (!primaryName) { model = buildPlaceholder(); scene.add(model); modelHeight = 1.85; captureBase(); return; }

    try {
      const gltf = await loadGLTF(`${assetBasePath}/${primaryName}.glb`);
      model = gltf.scene;
      model.traverse(n => { if (n.isMesh) { n.castShadow = true; n.receiveShadow = true; } });
      scene.add(model);
      mixer     = new THREE.AnimationMixer(model);
      boneNames = getBoneNames(model);

      // The primary GLB's own embedded clip — keep ALL tracks (it's the native
      // animation for this exact mesh, so position tracks are correct here).
      if (gltf.animations[0]) {
        clips[primaryName] = gltf.animations[0];
      }
    } catch (e) {
      console.warn('[char] primary load failed:', e);
      model = buildPlaceholder(); scene.add(model);
    }

    // Scale to ~1.85m tall, center on X/Z, sit on Y=0
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const scale = 1.85 / Math.max(size.y, 0.01);
    model.scale.setScalar(scale);
    model.position.set(-center.x, -box.min.y, -center.z);

    // Stash for loadAnimationClips
    model.userData._animNames = animNames;
    model.userData._primaryName = primaryName;

    // Model is normalized to ~1.85m tall above; capture the resting transform
    // as the breathing/look-at baseline.
    modelHeight = 1.85;
    captureBase();
  }

  // ── Load remaining animation clips in background ────────────────────────────
  async function loadAnimationClips() {
    if (!assetBasePath || !mixer || !model) return;

    const animNames   = model.userData._animNames || [];
    const primaryName = model.userData._primaryName;

    // Play the primary (idle) clip immediately — it's already loaded.
    if (clips[primaryName]) {
      playAnim(primaryName, { loop: true });
    }

    // Load the rest in parallel. Each clip binds to our skeleton by bone name.
    const others = animNames.filter(n => n !== primaryName);
    const results = await Promise.allSettled(
      others.map(name =>
        loadGLTF(`${assetBasePath}/${name}.glb`)
          .then(g => ({ name, clip: g.animations[0] ?? null }))
      )
    );

    for (const r of results) {
      if (r.status !== 'fulfilled' || !r.value.clip) continue;
      clips[r.value.name] = r.value.clip;
    }
  }

  // ── Particle system ────────────────────────────────────────────────────────
  const particles = (() => {
    const count = 100;
    const geo = new THREE.BufferGeometry();
    const pos = new Float32Array(count * 3);
    const vel = [];
    for (let i = 0; i < count; i++) {
      const a = Math.random() * Math.PI * 2, r = 0.38 + Math.random() * 0.3;
      pos[i*3] = Math.cos(a)*r; pos[i*3+1] = Math.random()*2.2; pos[i*3+2] = Math.sin(a)*r;
      vel.push({ a, r, speed: 0.4 + Math.random(), vy: 0.25 + Math.random() * 0.5 });
    }
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const mat = new THREE.PointsMaterial({ color: 0xff2222, size: 0.045, transparent: true, opacity: 0, depthWrite: false });
    const pts = new THREE.Points(geo, mat);
    scene.add(pts);
    return { pts, vel, geo, mat, active: false };
  })();

  function setParticles(active) {
    particles.active = active;
    if (!active) particles.mat.opacity = 0;
  }

  // ── Animation control ──────────────────────────────────────────────────────
  function playAnim(name, { loop = true, once = false, fade = 0.25 } = {}) {
    if (!mixer) return;
    // Animations are OFF by default — characters load in their static bind pose
    // (clean T-pose). Add ?anim=1 to the URL to enable playback.
    if (!new URLSearchParams(location.search).has('anim')) return;
    // Resolve clip — fallback chain: exact name → 'idle' → first available
    const clip = clips[name] ?? clips['idle'] ?? Object.values(clips)[0];
    if (!clip) return;

    const action = mixer.clipAction(clip);
    action.setLoop(once ? THREE.LoopOnce : THREE.LoopRepeat);
    action.clampWhenFinished = once;

    if (currentAction && currentAction !== action) {
      currentAction.fadeOut(fade);
    }
    action.reset().fadeIn(fade).play();
    currentAction = action;

    if (once) {
      const handler = e => {
        if (e.action !== action) return;
        mixer.removeEventListener('finished', handler);
        playAnim(personalityIdle in clips ? personalityIdle : 'idle', { loop: true });
      };
      mixer.addEventListener('finished', handler);
    }
  }

  // ── Resize ─────────────────────────────────────────────────────────────────
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / Math.max(h, 1);
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(canvas);
  resize();

  // ── Update loop ─────────────────────────────────────────────────────────────
  const clock = new THREE.Clock();
  function update() {
    const dt = clock.getDelta();
    if (mixer) mixer.update(dt);
    controls.update();

    elapsed += dt;

    // ── "Alive" idle: breathing bob/scale/sway + pointer look-at ──────────────
    // Applied relative to the captured base transform so it never drifts. When a
    // mixer clip is actively playing we damp the breathing way down (and skip the
    // scale/sway) so it layers gently on top of the skeletal animation instead of
    // fighting it — the rig already moves, we just keep the root subtly alive.
    if (model && baseSet && !materializing) {
      const clipActive = !!(currentAction && currentAction.isRunning && currentAction.isRunning());
      // Greet (procedural nod + small hop) takes priority over the idle bob.
      let greetBobY = 0;
      let greetRotX = 0;
      if (greetT > 0) {
        greetT = Math.max(0, greetT - dt);
        const prog = greetDur > 0 ? 1 - greetT / greetDur : 1; // 0 → 1
        // Two quick bobs over the duration + a forward nod that eases out.
        const env = Math.sin(Math.PI * Math.min(prog, 1)); // rise then fall
        greetBobY = Math.sin(prog * Math.PI * 4) * 0.06 * modelHeight * env;
        greetRotX = Math.sin(prog * Math.PI * 2) * 0.18 * env;
      }

      if (breathingOn || greetT > 0) {
        // Damp idle breathing while a real clip plays (apply only a small bob).
        const damp     = clipActive ? 0.25 : 1.0;
        const bobAmp   = 0.015 * modelHeight * damp;          // ~1.5% of height
        const bob      = Math.sin(elapsed * 1.6) * bobAmp;
        model.position.y = basePos.y + bob + greetBobY;

        if (!clipActive && greetT <= 0) {
          // Subtle scale breathing (~0.5%) and gentle Y sway — only when no clip
          // is driving the skeleton, to avoid a doubled / broken look.
          const breath = 1 + Math.sin(elapsed * 1.6) * 0.005;
          model.scale.set(baseScale.x * breath, baseScale.y * breath, baseScale.z * breath);
        } else {
          model.scale.copy(baseScale);
        }
        // Nod from greet (pitch) layered on top of base X rotation.
        model.rotation.x = greetRotX;
      } else {
        // Breathing off: settle back to base.
        model.position.y = basePos.y;
        model.scale.copy(baseScale);
        model.rotation.x = 0;
      }

      // Look-at-pointer yaw + a faint idle sway, smoothly damped toward target.
      const swayYaw = (breathingOn && !clipActive && greetT <= 0)
        ? Math.sin(elapsed * 0.6) * 0.03 : 0;
      const desiredY = baseRotY + (lookAtOn ? targetYaw : 0) + swayYaw;
      // Exponential damping (frame-rate independent-ish).
      const k = 1 - Math.exp(-6 * dt);
      model.rotation.y += (desiredY - model.rotation.y) * k;
    }

    // Particle animation
    if (particles.active) {
      particles.mat.opacity = Math.min(particles.mat.opacity + dt * 2.5, 0.75);
    } else {
      particles.mat.opacity = Math.max(particles.mat.opacity - dt * 4, 0);
    }
    if (particles.mat.opacity > 0.01) {
      const p = particles.geo.attributes.position.array;
      for (let i = 0; i < particles.vel.length; i++) {
        const v = particles.vel[i];
        v.a += v.speed * dt * 0.6;
        p[i*3]   = Math.cos(v.a) * v.r;
        p[i*3+1] = (p[i*3+1] + v.vy * dt) % 2.4;
        p[i*3+2] = Math.sin(v.a) * v.r;
      }
      particles.geo.attributes.position.needsUpdate = true;
    }

    renderer.render(scene, camera);
  }

  // ── "Alive" idle controls ───────────────────────────────────────────────────
  function setBreathing(on) {
    breathingOn = !!on;
  }

  // Pointer-driven look-at. We keep the handler referenced so dispose() can
  // remove it; it only updates targetYaw — the damping happens in update().
  const YAW_RANGE = 0.35; // ±0.35 rad clamp
  function onPointerMove(e) {
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    // Normalize pointer X to [-1, 1] across the canvas.
    const nx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    targetYaw = Math.max(-YAW_RANGE, Math.min(YAW_RANGE, nx * YAW_RANGE));
  }
  let pointerBound = false;
  function lookAtPointer(on) {
    lookAtOn = !!on;
    if (lookAtOn && !pointerBound) {
      canvas.addEventListener('pointermove', onPointerMove);
      pointerBound = true;
    } else if (!lookAtOn && pointerBound) {
      canvas.removeEventListener('pointermove', onPointerMove);
      pointerBound = false;
      targetYaw = 0; // update() eases the model back to forward
    }
  }

  // Intro "coming into being" — scale + opacity ease-out, REGARDLESS of ?anim.
  function materialize(durationMs = 1200) {
    if (!model || !baseSet) return Promise.resolve();
    // Collect materials we toggle so we can restore opaque ones at the end.
    const touched = []; // { mat, wasTransparent, baseOpacity }
    model.traverse(n => {
      if (!n.isMesh || !n.material) return;
      const mats = Array.isArray(n.material) ? n.material : [n.material];
      for (const mat of mats) {
        touched.push({ mat, wasTransparent: mat.transparent, baseOpacity: mat.opacity });
        mat.transparent = true;
      }
    });

    materializing = true;
    return new Promise(resolve => {
      const start = performance.now();
      const dur = Math.max(1, durationMs);
      const fromScale = 0.2;
      const tick = () => {
        const t = Math.min(1, (performance.now() - start) / dur);
        const e = 1 - Math.pow(1 - t, 3);          // ease-out cubic
        const s = fromScale + (1 - fromScale) * e;
        model.scale.set(baseScale.x * s, baseScale.y * s, baseScale.z * s);
        for (const r of touched) r.mat.opacity = r.baseOpacity * e;
        if (t < 1) { requestAnimationFrame(tick); return; }
        // Restore final state.
        model.scale.copy(baseScale);
        for (const r of touched) {
          r.mat.opacity = r.baseOpacity;
          r.mat.transparent = r.wasTransparent;
        }
        materializing = false;
        resolve();
      };
      requestAnimationFrame(tick);
    });
  }

  // Friendly greeting at the "birth" moment. Bypasses the ?anim gate.
  function greet() {
    if (!model) return Promise.resolve();
    if (mixer && clips['wave']) {
      // Play the wave clip once directly (bypassing playAnim's ?anim gate).
      return new Promise(resolve => {
        const action = mixer.clipAction(clips['wave']);
        action.setLoop(THREE.LoopOnce);
        action.clampWhenFinished = true;
        if (currentAction && currentAction !== action) currentAction.fadeOut(0.2);
        action.reset().fadeIn(0.2).play();
        currentAction = action;
        let done = false;
        const finish = () => {
          if (done) return; done = true;
          mixer.removeEventListener('finished', handler);
          resolve();
        };
        const handler = ev => { if (ev.action === action) finish(); };
        mixer.addEventListener('finished', handler);
        setTimeout(finish, 1800); // fallback
      });
    }
    // Procedural nod + small hop over ~1.2s.
    greetDur = 1.2;
    greetT = greetDur;
    return new Promise(resolve => setTimeout(resolve, 1200));
  }

  // ── Boot ───────────────────────────────────────────────────────────────────
  await loadBase();
  // loadAnimationClips plays the primary idle as soon as the model is ready,
  // then loads the rest of the clips in the background.
  loadAnimationClips();   // intentionally not awaited

  return {
    playAnim, setParticles, update,
    setBreathing, materialize, greet, lookAtPointer,
    dispose() {
      ro.disconnect();
      if (pointerBound) {
        canvas.removeEventListener('pointermove', onPointerMove);
        pointerBound = false;
      }
      renderer.dispose();
      if (mixer) mixer.stopAllAction();
    }
  };
}
