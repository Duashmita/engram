"""
Generate 3D characters for engram-chars presets using Meshy Gen-2.

Pipeline per character:
  1. Claude Haiku writes a personality-driven appearance description
  2. Meshy text-to-3D preview  (~400s)
  3. Meshy refine              (~200s, adds textures)
  4. Meshy rig                 (~120s, adds skeleton)
  5. Submit ALL animation tasks at once, then poll them in parallel (~60-90s)
  6. Save rigged base.glb + per-clip animation GLBs + manifest.json

Key design choices:
  - Parallel animation submission: all clips submitted at once, polled together.
    Goes from 18 × 30s = 9 min → 1-2 min for all clips.
  - Resumability: state.json records task IDs at each step. Re-running the
    script resumes from the last completed step.
  - Per-preset animation sets: each OCEAN profile gets animations that match
    its personality (e.g. guard gets crouching idle, merchant gets wave idle).
  - Progressive: base.glb saved as soon as rigging completes so the browser
    can display the character while animations download.

Usage:
    python scripts/generate_characters.py [--preset guard] [--all] [--desc-only]

Requires in .env:
    ANTHROPIC_API_KEY
    MESHY_API_KEY
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from engram.config import ANTHROPIC_API_KEY, MESHY_API_KEY
from engram.llm.client import AnthropicClient
from engram.presets import PRESETS, get_preset

MESHY_BASE = "https://api.meshy.ai"
OUTPUT_DIR = os.path.join(_REPO_ROOT, "docs", "assets", "characters")

# ---------------------------------------------------------------------------
# Per-preset animation sets
# Each preset downloads a curated set of Meshy action IDs suited to its
# personality. The same action_id values may appear across presets (that's
# fine — each rig task produces its own version).
# ---------------------------------------------------------------------------

# Full shared library — superset of all presets.
_ALL_ANIMS = {
    "idle":         0,    # Standard upright idle
    "idle_cross":   3,    # Arms-crossed idle (reserved/low-A)
    "crouching":    56,   # Crouching idle (defensive, high-N)
    "looking":      5,    # Looking around (curious, high-O)
    "talking":      12,   # Talking with gestures
    "thinking":     8,    # Thinking, hand to chin
    "nodding":      14,   # Nodding yes
    "shaking":      15,   # Shaking head no
    "wave":         17,   # Wave greeting (warm, high-E)
    "laughing":     23,   # Laughing (high-E/A)
    "clapping":     20,   # Clapping (positive)
    "scared":       22,   # Scared / alarmed
    "surprised":    21,   # Surprised
    "crying":       24,   # Crying
    "combat":       60,   # Combat idle stance
    "punch":        64,   # Punch attack
    "block":        67,   # Block / defend
    "getting_hit":  68,   # Getting hit (light)
    "dying":        72,   # Fall / die
}

# Per-preset curated sets.
PRESET_ANIMATIONS: dict[str, dict[str, int]] = {
    "guard": {   # high-N (0.9), low-A (0.2) — paranoid, defensive
        "idle":        56,   # crouching/guarded idle
        "talking":     12,
        "thinking":    8,
        "looking":     5,
        "scared":      22,
        "surprised":   21,
        "shaking":     15,
        "combat":      60,
        "getting_hit": 68,
        "dying":       72,
    },
    "merchant": {  # high-E (0.9), high-A (0.8) — warm, expressive
        "idle":        17,   # wave/greeting idle
        "talking":     12,
        "thinking":    8,
        "laughing":    23,
        "clapping":    20,
        "nodding":     14,
        "wave":        17,
        "surprised":   21,
        "getting_hit": 68,
        "dying":       72,
    },
    "clerk": {    # low-O (0.1), high-C (0.9) — rigid, precise
        "idle":        0,    # strict upright idle
        "talking":     12,
        "thinking":    8,
        "nodding":     14,
        "shaking":     15,
        "getting_hit": 68,
        "dying":       72,
    },
    "jeanie": {  # high-N (0.85), high-C (0.85) — anxious, precise
        "idle":        56,   # slightly defensive
        "talking":     12,
        "thinking":    8,
        "scared":      22,
        "nodding":     14,
        "shaking":     15,
        "getting_hit": 68,
        "dying":       72,
    },
    "maya": {    # high-O (0.9), high-E (0.85), high-A (0.8) — expressive, open
        "idle":        5,    # curious, looking around
        "talking":     12,
        "thinking":    8,
        "laughing":    23,
        "clapping":    20,
        "wave":        17,
        "surprised":   21,
        "getting_hit": 68,
        "dying":       72,
    },
    "hale": {    # high-C (0.85), low-A (0.2), low-N (0.25) — blunt, authoritative
        "idle":        3,    # arms-crossed authoritative idle
        "talking":     12,
        "thinking":    8,
        "nodding":     14,
        "shaking":     15,
        "combat":      60,
        "getting_hit": 68,
        "dying":       72,
    },
}

# Fall back to a generic set for unknown presets.
_DEFAULT_ANIMATIONS = {
    "idle": 0, "talking": 12, "thinking": 8, "getting_hit": 68, "dying": 72,
}


# ---------------------------------------------------------------------------
# Meshy API helpers
# ---------------------------------------------------------------------------

def _headers(api_key: str) -> dict:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _post(url: str, api_key: str, payload: dict, timeout: int = 90, retries: int = 4) -> dict:
    """POST a task-submission request with retries on timeout/5xx.

    Meshy's submission endpoints occasionally take >30s to respond. We use a
    generous timeout and exponential backoff so a slow accept doesn't kill the
    whole run (which would otherwise waste the completed preview/refine work).
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
            # Some 500s are non-transient (e.g. rigging pose-estimation failure on a
            # mesh the auto-rigger can't parse). Don't waste retries on those.
            if resp.status_code >= 500:
                body = resp.text[:160]
                if "pose estimation" in body.lower():
                    raise RuntimeError(f"non-retryable rigging failure: {body}")
                raise requests.exceptions.HTTPError(f"{resp.status_code}: {body}")
            resp.raise_for_status()
            return resp.json()
        except RuntimeError:
            raise  # non-retryable — surface immediately
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError,
                requests.exceptions.HTTPError) as exc:
            last_exc = exc
            wait = 5 * (2 ** attempt)
            print(f"  [retry] POST {url.split('/')[-1]} attempt {attempt+1}/{retries} failed "
                  f"({type(exc).__name__}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"POST failed after {retries} attempts: {last_exc}")


def _poll(api_key: str, url: str, label: str = "", interval: int = 8, max_tries: int = 90):
    for _ in range(max_tries):
        time.sleep(interval)
        data = requests.get(url, headers=_headers(api_key), timeout=15).json()
        state = data.get("status", "")
        pct = data.get("progress", 0)
        print(f"  [{label}] {state} {pct}%", end="\r", flush=True)
        if state == "SUCCEEDED":
            print()
            return data
        if state in ("FAILED", "EXPIRED"):
            raise RuntimeError(f"Task failed: {data}")
    raise TimeoutError(f"Task timed out after {max_tries * interval}s")


def _extract_anim_url(data: dict) -> str | None:
    """Defensively extract the animation GLB URL from various Meshy response shapes."""
    result = data.get("result")
    if isinstance(result, dict):
        for key in ("model_url", "animation_glb_url", "glb_url", "rigged_character_glb_url"):
            if result.get(key):
                return result[key]
    if isinstance(result, str) and result.startswith("http"):
        return result
    for key in ("model_url", "animation_url", "glb_url"):
        if data.get(key) and data[key].startswith("http"):
            return data[key]
    return None


def _download(url: str, path: str) -> None:
    resp = requests.get(url, timeout=120, stream=True)
    resp.raise_for_status()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        for chunk in resp.iter_content(65536):
            f.write(chunk)


# ---------------------------------------------------------------------------
# Generation pipeline steps
# ---------------------------------------------------------------------------

def step_description(client: AnthropicClient, config) -> str:
    """Step 1 — ask Claude to write a personality-expressive appearance description."""
    o = config.profile
    # Map trait values to strong prose descriptors
    def trait_prose(trait, val):
        if trait == "N":
            if val >= 0.7: return "visibly anxious, guarded, tense — braced for threat"
            if val <= 0.3: return "calm and relaxed, loose posture, at ease"
        if trait == "E":
            if val >= 0.7: return "open posture, arms wide, physically expressive"
            if val <= 0.3: return "closed, contained, arms close to body"
        if trait == "A":
            if val >= 0.7: return "warm expression, approachable, soft features"
            if val <= 0.3: return "hard expression, suspicious, jaw clenched"
        if trait == "O":
            if val >= 0.7: return "curious tilt to the head, alive with interest"
            if val <= 0.3: return "rigid, conventional bearing, nothing about them surprises"
        if trait == "C":
            if val >= 0.7: return "precisely upright, impeccable posture, controlled"
            if val <= 0.3: return "slouched, casual, unbothered by appearances"
        return ""

    traits = " | ".join(
        f"{t}: {trait_prose(t, getattr(o, t))}"
        for t in ("O", "C", "E", "A", "N")
        if trait_prose(t, getattr(o, t))
    )

    prompt = f"""Write a 2–3 sentence 3D character appearance description for a game NPC.
The description will be sent directly to a 3D generation API (Meshy).

Character: {config.name}
Role/persona: {config.persona[:200]}
Personality (OCEAN Big Five, 0–1 scale):
  {traits}

Rules:
- Focus exclusively on BODY SHAPE, POSTURE, FACIAL STRUCTURE, and EXPRESSION.
- Let the personality radiate through body language above all else.
- Do NOT mention colors or skin tone.
- Be specific and evocative — avoid generic phrases like "weathered" or "rugged".
- End with exactly: "T-pose, humanoid, game character, realistic proportions."

Write only the description, nothing else."""

    return client.generate(prompt, max_tokens=220).strip()


def step_preview(api_key: str, description: str) -> str:
    """Step 2 — Meshy text-to-3D preview. Returns preview task_id."""
    data = _post(f"{MESHY_BASE}/openapi/v2/text-to-3d", api_key,
                 {"mode": "preview", "prompt": description,
                  "art_style": "realistic", "should_remesh": True})
    task_id = data["result"]
    print(f"  [preview] task={task_id}")
    _poll(api_key, f"{MESHY_BASE}/openapi/v2/text-to-3d/{task_id}", "preview")
    return task_id


def step_refine(api_key: str, preview_id: str) -> tuple[str, str]:
    """Step 3 — Meshy refine (textures). Returns (refine_task_id, glb_url)."""
    data = _post(f"{MESHY_BASE}/openapi/v2/text-to-3d", api_key,
                 {"mode": "refine", "preview_task_id": preview_id})
    task_id = data["result"]
    print(f"  [refine] task={task_id}")
    result = _poll(api_key, f"{MESHY_BASE}/openapi/v2/text-to-3d/{task_id}", "refine")
    return task_id, result["model_urls"]["glb"]


def step_rig(api_key: str, refine_task_id: str) -> tuple[str, str]:
    """Step 4 — Meshy auto-rig. Returns (rig_task_id, rigged_glb_url)."""
    data = _post(f"{MESHY_BASE}/openapi/v1/rigging", api_key,
                 {"input_task_id": refine_task_id})
    task_id = data["result"]
    print(f"  [rig] task={task_id}")
    result = _poll(api_key, f"{MESHY_BASE}/openapi/v1/rigging/{task_id}", "rig")
    return task_id, result["result"]["rigged_character_glb_url"]


def step_animations(api_key: str, rig_task_id: str, anim_map: dict[str, int],
                    out_dir: str) -> dict[str, str]:
    """
    Step 5 — Submit ALL animation tasks at once, then poll them in parallel.

    Instead of: submit → wait → submit → wait (18 × ~30s = 9 min)
    We do:      submit all → wait once for all (~60-90s total)

    Returns {name: relative_path} for each successfully downloaded clip.
    """
    headers = _headers(api_key)

    # Submit all tasks at once (resilient POST with retries)
    pending: dict[str, str] = {}   # name → anim_task_id
    for name, action_id in anim_map.items():
        try:
            data = _post(f"{MESHY_BASE}/openapi/v1/animations", api_key,
                         {"rig_task_id": rig_task_id, "action_id": action_id})
            pending[name] = data["result"]
            print(f"  [anim] submitted {name} (id={action_id}) → task={pending[name]}")
        except Exception as exc:
            print(f"  [anim] submit error {name}: {exc}")

    if not pending:
        print("  [anim] no animation tasks submitted")
        return {}

    print(f"  [anim] {len(pending)} tasks submitted — polling...")

    # Poll all tasks together until all complete
    completed: dict[str, str] = {}   # name → download_url
    failed: set[str] = set()
    max_rounds = 60
    for round_n in range(max_rounds):
        time.sleep(10)
        done_this_round: list[str] = []
        for name, task_id in pending.items():
            try:
                data = requests.get(
                    f"{MESHY_BASE}/openapi/v1/animations/{task_id}",
                    headers=headers, timeout=15
                ).json()
                state = data.get("status", "")
                if state == "SUCCEEDED":
                    url = _extract_anim_url(data)
                    if url:
                        completed[name] = url
                    else:
                        print(f"  [anim] {name} succeeded but no URL: {json.dumps(data)[:200]}")
                    done_this_round.append(name)
                elif state in ("FAILED", "EXPIRED"):
                    failed.add(name)
                    done_this_round.append(name)
            except Exception as exc:
                print(f"  [anim] poll error {name}: {exc}")

        for name in done_this_round:
            del pending[name]

        n_done = len(completed) + len(failed)
        n_total = n_done + len(pending)
        print(f"  [anim] round {round_n + 1}: {len(completed)} ok, {len(failed)} failed, {len(pending)} pending / {n_total} total", end="\r", flush=True)

        if not pending:
            break

    print()

    # Download completed clips in parallel
    def _dl_clip(name_url: tuple[str, str]) -> tuple[str, str | None]:
        name, url = name_url
        path = os.path.join(out_dir, f"{name}.glb")
        try:
            _download(url, path)
            return name, path
        except Exception as exc:
            print(f"  [anim] download failed {name}: {exc}")
            return name, None

    preset_key = os.path.basename(out_dir)
    result_paths: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(_dl_clip, (n, u)): n for n, u in completed.items()}
        for future in as_completed(futures):
            name, path = future.result()
            if path:
                result_paths[name] = f"assets/characters/{preset_key}/{name}.glb"
                print(f"  [anim] saved {name}.glb")

    return result_paths


# ---------------------------------------------------------------------------
# State management (resumability)
# ---------------------------------------------------------------------------

def _load_state(out_dir: str) -> dict:
    path = os.path.join(out_dir, "state.json")
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return {}


def _save_state(out_dir: str, state: dict) -> None:
    with open(os.path.join(out_dir, "state.json"), "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_for_preset(preset_key: str, client: AnthropicClient, api_key: str,
                        force: bool = False) -> None:
    config = get_preset(preset_key)
    out_dir = os.path.join(OUTPUT_DIR, preset_key)
    os.makedirs(out_dir, exist_ok=True)

    state = {} if force else _load_state(out_dir)
    anim_map = PRESET_ANIMATIONS.get(preset_key, _DEFAULT_ANIMATIONS)

    print(f"\n{'='*60}")
    print(f"  {preset_key} — {config.name}")
    print(f"  O={config.profile.O} C={config.profile.C} E={config.profile.E} A={config.profile.A} N={config.profile.N}")
    print(f"{'='*60}")

    # Step 1: description
    if not state.get("description"):
        desc = step_description(client, config)
        state["description"] = desc
        _save_state(out_dir, state)
    print(f"  [desc] {state['description'][:120]}...")

    # Step 2: preview
    if not state.get("preview_id"):
        state["preview_id"] = step_preview(api_key, state["description"])
        _save_state(out_dir, state)

    # Step 3: refine
    if not state.get("refine_id"):
        state["refine_id"], state["refine_glb_url"] = step_refine(api_key, state["preview_id"])
        _save_state(out_dir, state)

    # Step 4: rig
    if not state.get("rig_id"):
        state["rig_id"], state["rigged_glb_url"] = step_rig(api_key, state["refine_id"])
        _save_state(out_dir, state)

    # Download rigged base GLB
    base_path = os.path.join(out_dir, "base.glb")
    if not os.path.exists(base_path) or force:
        print("  [download] base.glb...")
        _download(state["rigged_glb_url"], base_path)
        print(f"  [saved] {base_path}")
    else:
        print("  [skip] base.glb already exists")

    # Step 5: animations (parallel submit + poll + download)
    if not state.get("animations_done"):
        anim_paths = step_animations(api_key, state["rig_id"], anim_map, out_dir)
        state["animation_paths"] = anim_paths
        state["animations_done"] = True
        _save_state(out_dir, state)
    else:
        print(f"  [skip] animations already done: {list(state.get('animation_paths', {}).keys())}")
        anim_paths = state.get("animation_paths", {})

    # Write manifest
    preset_key_str = preset_key
    manifest = {
        "preset":      preset_key,
        "name":        config.name,
        "description": state["description"],
        "ocean":       {"O": config.profile.O, "C": config.profile.C,
                        "E": config.profile.E, "A": config.profile.A, "N": config.profile.N},
        "base":        f"assets/characters/{preset_key}/base.glb",
        "animations":  anim_paths,
        "idle_anim":   next(
            (n for n, aid in anim_map.items() if aid == anim_map.get("idle") and aid != 0),
            "idle"
        ),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [manifest] written — {len(anim_paths)} animations")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Meshy characters for engram-chars")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), help="Generate one preset")
    parser.add_argument("--all", action="store_true", help="Generate all presets")
    parser.add_argument("--desc-only", action="store_true", help="Print descriptions only (no API cost)")
    parser.add_argument("--force", action="store_true", help="Ignore saved state, regenerate everything")
    args = parser.parse_args()

    if not ANTHROPIC_API_KEY:
        print("error: ANTHROPIC_API_KEY not set", file=sys.stderr); sys.exit(1)

    client = AnthropicClient(api_key=ANTHROPIC_API_KEY)

    if args.desc_only:
        targets = list(PRESETS.keys()) if args.all else [args.preset or "guard"]
        for key in targets:
            cfg = get_preset(key)
            desc = step_description(client, cfg)
            print(f"\n[{key}] {cfg.name}")
            print(f"  {desc}\n")
        return

    if not MESHY_API_KEY:
        print("error: MESHY_API_KEY not set", file=sys.stderr); sys.exit(1)

    targets = list(PRESETS.keys()) if args.all else [args.preset or "guard"]
    for key in targets:
        generate_for_preset(key, client, MESHY_API_KEY, force=args.force)

    print("\n\nAll done.")


if __name__ == "__main__":
    main()
