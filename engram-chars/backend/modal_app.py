"""
Engram live-demo backend.

A FastAPI app that runs the Engram pipeline per visitor and streams pipeline
events back as Server-Sent Events. Wrapped as a Modal ASGI app so that

    modal deploy backend/modal_app.py

ships it. For local development without Modal:

    uvicorn backend.modal_app:api --port 8000 --reload

The FastAPI app is exported as the module-level name ``api`` so it can be
imported by both the Modal wrapper and uvicorn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import requests

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse


# ---------------------------------------------------------------------------
# Path setup, make sure ``engram`` is importable in both Modal and local dev
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)

# Local dev: src is at <repo>/src.
# Modal: deploy file lives at /root/modal_app.py while the repo (mounted via
# add_local_dir) is at /root/engram, so src is at /root/engram/src. Both paths
# are checked here at module load, the engram imports below MUST resolve
# before the Modal runner inspects the FastAPI app.
for _candidate in (os.path.join(REPO_ROOT, "src"), "/root/engram/src"):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from engram import config as engram_config  # noqa: E402
from engram.config import MESHY_API_KEY  # noqa: E402
from engram.llm.client import AnthropicClient as GeminiClient  # noqa: E402
from engram.models import NPCConfig, OCEANProfile  # noqa: E402
from engram.npc import NPCAgent  # noqa: E402
from engram.observability import bus  # noqa: E402
from engram.presets import PRESETS, get_preset  # noqa: E402

# SQLite persistence layer (backend/db.py). Same dual-path dance as src above:
# next to this file in local dev, under /root/engram/backend in the container.
for _candidate in (_THIS_DIR, "/root/engram/backend"):
    if os.path.isdir(_candidate) and _candidate not in sys.path:
        sys.path.insert(0, _candidate)
import db as engram_db  # noqa: E402


log = logging.getLogger("engram.backend")
if not log.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SESSION_TTL_S = 3600
HARD_TURN_CAP = 30
# Rate limits apply to the shared-key path (no BYOK). The onboarding wizard fires
# several calls per character (infer_character, greeting, start, …), so
# keep these generous, the old 5/min tripped mid-wizard and surfaced as
# "personality model unreachable".
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "120"))
RATE_LIMIT_PER_DAY = int(os.environ.get("RATE_LIMIT_PER_DAY", "2000"))

_SESSION_BASE_DIR = "/tmp/engram_sessions"

# ---------------------------------------------------------------------------
# Device-persistent memory store
# ---------------------------------------------------------------------------
# Backed by SQLite (backend/db.py); on Modal the DB lives on a persistent
# Volume mounted at /data (wired in the Modal block at the bottom). The
# in-process dict is a last-resort fallback when the DB is unavailable.
# Keys are "<device_id>:<npc_id>".
_DEVICE_MEM: dict = {}


def _split_device_key(key: str) -> tuple[str, str]:
    device_id, _, npc_id = key.partition(":")
    return npc_id, device_id


def _legacy_device_get(key: str) -> dict | None:
    """Read from the pre-SQLite modal.Dict store. Overridden in the Modal
    block when that Dict is reachable; no-op everywhere else."""
    return None


def _device_get(key: str) -> dict | None:
    npc_id, device_id = _split_device_key(key)
    try:
        saved = engram_db.load_npc_snapshot(npc_id, device_id)
    except Exception as exc:  # noqa: BLE001
        log.warning("db snapshot load failed key=%s…: %s", key[:12], exc)
        saved = None
    if saved is not None:
        return saved
    # One-time lazy migration from the legacy modal.Dict store.
    legacy = _legacy_device_get(key)
    if legacy:
        try:
            engram_db.save_npc_snapshot(npc_id, device_id, legacy)
            log.info("migrated legacy device memory key=%s…", key[:12])
        except Exception:  # noqa: BLE001
            pass
        return legacy
    return _DEVICE_MEM.get(key)


def _device_put(key: str, value: dict) -> None:
    npc_id, device_id = _split_device_key(key)
    try:
        engram_db.save_npc_snapshot(npc_id, device_id, value)
    except Exception as exc:  # noqa: BLE001
        log.warning("db snapshot save failed key=%s…: %s", key[:12], exc)
        _DEVICE_MEM[key] = value


def _restore_device_memory(device_id: str, npc_id: str, data_dir_root: str) -> bool:
    """Write saved files into the session sandbox before NPCAgent construction.

    Returns True if memory was found and restored; False if this is a new device.
    """
    saved = _device_get(f"{device_id}:{npc_id}")
    if not saved:
        return False
    npc_dir = os.path.join(data_dir_root, npc_id)
    os.makedirs(npc_dir, exist_ok=True)
    try:
        for filename, field in (
            ("memories.json", "memories"),
            ("longterm.json", "longterm"),
        ):
            if field in saved:
                with open(os.path.join(npc_dir, filename), "w", encoding="utf-8") as f:
                    json.dump(saved[field], f, indent=2, ensure_ascii=False)
        if "keystore" in saved:
            with open(os.path.join(npc_dir, "keystore.pl"), "w", encoding="utf-8") as f:
                f.write(saved["keystore"])
        if "state" in saved:
            # Restore state but reset history, each session starts a fresh
            # conversation; long-term identity comes from memories, not raw logs.
            state = dict(saved["state"])
            state["history"] = []
            state["turn_count"] = 0
            with open(os.path.join(npc_dir, "state.json"), "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except Exception as exc:
        log.warning("device restore failed device=%s npc=%s: %s", device_id[:8], npc_id, exc)
        return False


def _save_device_memory(sess: "_Session") -> None:
    """Snapshot the NPC's current memory files to the device store."""
    if not sess.device_id:
        return
    # Persist in-memory state to disk first.
    try:
        sess.agent.save_state()
    except Exception as exc:
        log.warning("save_state failed sid=? device=%s: %s", sess.device_id[:8], exc)

    npc_dir = os.path.join(sess.data_dir, sess.npc_id)
    key = f"{sess.device_id}:{sess.npc_id}"
    data: dict = {}
    for filename, field in (
        ("memories.json", "memories"),
        ("longterm.json", "longterm"),
    ):
        path = os.path.join(npc_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    data[field] = json.load(f)
            except Exception:
                pass
    keystore_path = os.path.join(npc_dir, "keystore.pl")
    if os.path.exists(keystore_path):
        try:
            with open(keystore_path, encoding="utf-8") as f:
                data["keystore"] = f.read()
        except Exception:
            pass
    state_path = os.path.join(npc_dir, "state.json")
    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as f:
                data["state"] = json.load(f)
        except Exception:
            pass
    if data:
        _device_put(key, data)
        log.info("device memory saved device=%s npc=%s", sess.device_id[:8], sess.npc_id)
        # Append to the attribution ledger so the NPC's accumulated memories
        # stay traceable to the speaker who was present when they formed.
        try:
            engram_db.append_memories(sess.npc_id, sess.device_id, [
                {
                    "id": m.get("id"),
                    "text": m.get("text"),
                    "source": m.get("source"),
                    "importance": (m.get("tags") or {}).get("importance"),
                    "tags": m.get("tags"),
                }
                for m in (data.get("memories") or []) if isinstance(m, dict)
            ])
        except Exception as exc:  # noqa: BLE001
            log.warning("memory ledger append failed npc=%s: %s", sess.npc_id, exc)


# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    npc_id: str
    agent: NPCAgent
    llm: GeminiClient
    data_dir: str
    device_id: Optional[str] = None
    turn_count: int = 0
    last_used_ts: float = field(default_factory=time.time)


SESSIONS: dict[str, _Session] = {}
# ip -> list[timestamps]; cleaned up inline by _check_rate.
_RATE: dict[str, list[float]] = {}


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StartReq(BaseModel):
    npc_id: str
    device_id: Optional[str] = None
    anthropic_key: Optional[str] = None
    gemini_key: Optional[str] = None  # kept for backwards compat; anthropic_key takes precedence
    ocean: Optional[dict] = None  # {"O": 0.0-1.0, "C": ..., "E": ..., "A": ..., "N": ...}
    # ── Custom character (onboarding wizard) ──────────────────────────────
    custom: Optional[bool] = False
    name: Optional[str] = None
    persona: Optional[str] = None
    backstory: Optional[list] = None   # list[str], seed memories
    facts: Optional[list] = None       # list[str], seed prolog facts


class TurnReq(BaseModel):
    session_id: str
    player_input: str
    anthropic_key: Optional[str] = None
    gemini_key: Optional[str] = None  # kept for backwards compat


# /end parses its body manually (beacons send text/plain), no model needed.


class InferOceanReq(BaseModel):
    qa: list  # [{"question": str, "answer": str}, ...]
    anthropic_key: Optional[str] = None


class InferCharacterReq(BaseModel):
    qa: list  # [{"question": str, "answer": str}, ...]
    name: Optional[str] = ""
    anthropic_key: Optional[str] = None


class AppearanceReq(BaseModel):
    name: str
    persona: Optional[str] = ""
    ocean: dict
    anthropic_key: Optional[str] = None


class GreetingReq(BaseModel):
    name: str
    persona: Optional[str] = ""
    ocean: dict
    anthropic_key: Optional[str] = None
    # When set, the greeting is recorded in that session's history so the
    # NPC knows it already spoke first.
    session_id: Optional[str] = None


class GenerateCharacterReq(BaseModel):
    name: str
    description: str


# ---------------------------------------------------------------------------
# Meshy text-to-3D background jobs
# ---------------------------------------------------------------------------
# Each job_id -> {"status": "running"|"done"|"error",
#                 "stage": "queued"|"preview"|"refine"|"rig"|"failed",
#                 "glb_url": <str|None>,   # latest best available model URL
#                 "progress": <int 0-100>,
#                 "error": <str|None>}

MESHY_BASE = "https://api.meshy.ai"

CHARACTER_JOBS: dict = {}
_CHARACTER_JOBS_LOCK = threading.Lock()

# Polling tuning.
_MESHY_POLL_INTERVAL_S = 8
_MESHY_STAGE_TIMEOUT_S = 20 * 60  # ~20 min cap per stage
_MESHY_POST_TIMEOUT_S = 60
_MESHY_POST_RETRIES = 2


def _meshy_headers() -> dict:
    return {
        "Authorization": f"Bearer {MESHY_API_KEY}",
        "Content-Type": "application/json",
    }


def _get_job(job_id: str) -> Optional[dict]:
    """Read a job snapshot under the lock (returns a copy)."""
    with _CHARACTER_JOBS_LOCK:
        job = CHARACTER_JOBS.get(job_id)
        return dict(job) if job is not None else None


def _update_job(job_id: str, **fields) -> None:
    """Mutate a job's fields under the lock."""
    with _CHARACTER_JOBS_LOCK:
        job = CHARACTER_JOBS.get(job_id)
        if job is not None:
            job.update(fields)


class _PoseEstimationError(Exception):
    """Non-fatal rigging failure (clothed/bulky mesh). Degrade to refined mesh."""


def _meshy_post(path: str, payload: dict) -> str:
    """POST to a Meshy create endpoint and return the result task_id.

    Retries on request timeouts, but never retries a 500 whose body mentions
    "pose estimation", that is a terminal rigging failure raised as
    _PoseEstimationError so the caller can degrade gracefully.
    """
    url = f"{MESHY_BASE}{path}"
    last_exc: Optional[Exception] = None
    for attempt in range(_MESHY_POST_RETRIES + 1):
        try:
            resp = requests.post(
                url, json=payload, headers=_meshy_headers(),
                timeout=_MESHY_POST_TIMEOUT_S,
            )
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            log.warning("meshy POST %s timed out (attempt %d)", path, attempt + 1)
            continue
        if resp.status_code == 500 and "pose estimation" in (resp.text or "").lower():
            raise _PoseEstimationError(resp.text[:200])
        resp.raise_for_status()
        data = resp.json()
        result = data.get("result")
        if not result:
            raise RuntimeError(f"meshy POST {path} returned no result: {data}")
        return result
    raise last_exc or RuntimeError(f"meshy POST {path} failed")


def _meshy_poll(path: str, extract) -> str:
    """Poll a Meshy GET endpoint until SUCCEEDED, returning a value from `extract`.

    `extract(data)` pulls the GLB url out of the SUCCEEDED payload. Raises on
    FAILED/EXPIRED or when the per-stage timeout is exceeded.
    """
    url = f"{MESHY_BASE}{path}"
    deadline = time.time() + _MESHY_STAGE_TIMEOUT_S
    while True:
        resp = requests.get(url, headers=_meshy_headers(), timeout=_MESHY_POST_TIMEOUT_S)
        if resp.status_code == 500 and "pose estimation" in (resp.text or "").lower():
            raise _PoseEstimationError(resp.text[:200])
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status")
        if status == "SUCCEEDED":
            glb = extract(data)
            if not glb:
                raise RuntimeError(f"meshy GET {path} succeeded but no glb url")
            return glb
        if status in ("FAILED", "EXPIRED"):
            raise RuntimeError(f"meshy task {status}: {str(data.get('task_error') or data)[:200]}")
        if time.time() >= deadline:
            raise RuntimeError(f"meshy GET {path} timed out (last status={status})")
        time.sleep(_MESHY_POLL_INTERVAL_S)


def _generate_character_worker(job_id: str, name: str, description: str) -> None:
    """Background pipeline: preview -> refine -> rig, updating the job each stage.

    Rig failures (including pose-estimation 500s) are non-fatal: the refined
    mesh is kept as the final result. Only preview/refine failures mark error.
    """
    refine_glb: Optional[str] = None
    try:
        # ── Preview ───────────────────────────────────────────────────────
        log.info("character job %s: preview start name=%s", job_id, name)
        preview_task_id = _meshy_post("/openapi/v2/text-to-3d", {
            "mode": "preview",
            "prompt": description,
            "art_style": "realistic",
            "should_remesh": True,
        })
        preview_glb = _meshy_poll(
            f"/openapi/v2/text-to-3d/{preview_task_id}",
            lambda d: (d.get("model_urls") or {}).get("glb"),
        )
        _update_job(job_id, stage="preview", glb_url=preview_glb, progress=33)
        log.info("character job %s: preview done", job_id)

        # ── Refine ────────────────────────────────────────────────────────
        refine_task_id = _meshy_post("/openapi/v2/text-to-3d", {
            "mode": "refine",
            "preview_task_id": preview_task_id,
        })
        refine_glb = _meshy_poll(
            f"/openapi/v2/text-to-3d/{refine_task_id}",
            lambda d: (d.get("model_urls") or {}).get("glb"),
        )
        _update_job(job_id, stage="refine", glb_url=refine_glb, progress=66)
        log.info("character job %s: refine done", job_id)

        # ── Rig (best-effort; degrade to refined mesh on failure) ─────────
        try:
            rig_task_id = _meshy_post("/openapi/v1/rigging", {
                "input_task_id": refine_task_id,
            })
            rigged_glb = _meshy_poll(
                f"/openapi/v1/rigging/{rig_task_id}",
                lambda d: (d.get("result") or {}).get("rigged_character_glb_url"),
            )
            _update_job(
                job_id, status="done", stage="rig",
                glb_url=rigged_glb, progress=100,
            )
            log.info("character job %s: rig done", job_id)
        except _PoseEstimationError as exc:
            log.info("character job %s: rig pose-estimation failure, keeping refined mesh: %s",
                     job_id, exc)
            _update_job(job_id, status="done", stage="refine", glb_url=refine_glb, progress=100)
        except Exception as exc:  # noqa: BLE001
            log.warning("character job %s: rig failed, keeping refined mesh: %s", job_id, exc)
            _update_job(job_id, status="done", stage="refine", glb_url=refine_glb, progress=100)

    except _PoseEstimationError as exc:
        # Pose-estimation surfaced before refine completed, only degrade if we
        # actually have a refined mesh; otherwise it's a genuine failure.
        if refine_glb:
            _update_job(job_id, status="done", stage="refine", glb_url=refine_glb, progress=100)
        else:
            log.warning("character job %s failed: %s", job_id, exc)
            _update_job(job_id, status="error", stage="failed", error=str(exc)[:200])
    except Exception as exc:  # noqa: BLE001
        log.exception("character job %s failed", job_id)
        _update_job(job_id, status="error", stage="failed", error=str(exc)[:200])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return (request.client.host if request.client else "") or "unknown"


def _slugify(name: str) -> str:
    """Lowercase slug; non-alphanumeric runs → underscores. Fallback custom_npc."""
    slug = "".join(c if c.isalnum() else "_" for c in (name or "").lower())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "custom_npc"


def _ocean_from_dict(ocean: Optional[dict]) -> dict[str, float]:
    """Parse an OCEAN dict into clamped trait floats; default 0.5 each."""
    ocean = ocean or {}
    out: dict[str, float] = {}
    for trait in ('O', 'C', 'E', 'A', 'N'):
        try:
            val = float(ocean.get(trait, 0.5))
        except (TypeError, ValueError):
            val = 0.5
        out[trait] = round(max(0.0, min(1.0, val)), 3)
    return out


def _purge_expired() -> None:
    """Drop sessions older than the TTL. Runs at the top of every request."""
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s.last_used_ts > SESSION_TTL_S]
    for sid in expired:
        s = SESSIONS.pop(sid, None)
        if s is not None:
            _save_device_memory(s)
            try:
                engram_db.record_session_end(sid, s.turn_count)
            except Exception:  # noqa: BLE001
                pass
            shutil.rmtree(s.data_dir, ignore_errors=True)
            log.info("session expired sid=%s npc=%s", sid, s.npc_id)


def _check_rate(ip: str) -> tuple[bool, int]:
    """Sliding-window per-IP rate limit. Returns (ok, retry_after_seconds)."""
    now = time.time()
    minute_ago = now - 60.0
    day_ago = now - 86400.0

    bucket = _RATE.get(ip, [])
    # Inline cleanup, drop anything older than the daily window.
    bucket = [t for t in bucket if t >= day_ago]
    in_minute = sum(1 for t in bucket if t >= minute_ago)

    if in_minute >= RATE_LIMIT_PER_MIN:
        oldest_in_min = min(t for t in bucket if t >= minute_ago)
        retry = max(1, int(60 - (now - oldest_in_min)) + 1)
        _RATE[ip] = bucket
        return False, retry

    if len(bucket) >= RATE_LIMIT_PER_DAY:
        oldest = min(bucket)
        retry = max(60, int(86400 - (now - oldest)) + 1)
        _RATE[ip] = bucket
        return False, retry

    bucket.append(now)
    _RATE[ip] = bucket
    return True, 0


def _resolve_llm(byok: Optional[str]) -> GeminiClient:
    """Pick the BYOK key if present, else the shared env key. 503 if neither."""
    key = (byok or os.environ.get("ANTHROPIC_API_KEY", "")).strip()
    if not key:
        raise HTTPException(503, "No Anthropic API key configured and none provided")
    try:
        return GeminiClient(api_key=key)
    except Exception as exc:  # noqa: BLE001
        log.warning("AnthropicClient init failed: %s", exc)
        raise HTTPException(503, f"Client init failed: {exc}")


def _make_data_dir(session_id: str, npc_id: str) -> str:
    """Per-session sandbox under /tmp. Reuses pre-baked preset data when present.

    Local dev: pre-baked data lives at <repo>/data/<npc_id>/.
    Modal container: the repo is mounted at /root/engram, so it lives at
    /root/engram/data/<npc_id>/. Try both, first match wins.
    """
    os.makedirs(_SESSION_BASE_DIR, exist_ok=True)
    dst = os.path.join(_SESSION_BASE_DIR, session_id)
    for src in (
        os.path.join(REPO_ROOT, "data", npc_id),
        f"/root/engram/data/{npc_id}",
    ):
        if os.path.isdir(src):
            # copytree won't overwrite existing dirs without dirs_exist_ok=True (Py 3.8+).
            shutil.copytree(src, dst, dirs_exist_ok=True)
            return dst
    os.makedirs(dst, exist_ok=True)
    return dst


def _serialize_memory(m) -> dict:
    """Compact memory shape for the frontend memory panel."""
    tags = getattr(m, "tags", None)
    importance = getattr(tags, "importance", None) if tags else None
    return {
        "id": getattr(m, "id", ""),
        "text": getattr(m, "text", ""),
        "source": getattr(m, "source", "backstory"),
        "importance": importance if importance is not None else 5,
    }


def _build_header(agent: NPCAgent) -> dict:
    """Mirror chat.py's _build_viz_header but populate from a live agent."""
    p = agent.config.profile
    mems = list(agent.memory_manager.all_memories)

    # Promote key memories now so the "Key memories" panel shows the character's
    # most salient backstory from the start (normally promotion happens at end of
    # session). Also surface seeded Prolog facts. Both load outside the event bus,
    # so without this the panel stays empty until a turn fires.
    ks = getattr(agent.memory_manager, "keystore", None)
    key_ids, facts = [], []
    if ks is not None:
        try:
            ks.update(mems, p)
            key_ids = [m.id for m in getattr(ks, "_key_memories", [])]
            facts = list(getattr(ks, "_fact_strings", []))
        except Exception as exc:  # noqa: BLE001
            log.warning("key-memory promotion at start failed: %s", exc)

    return {
        "npc_id": agent.config.npc_id,
        "npc_name": agent.config.name,
        "persona": agent.config.persona,
        "baseline_ocean": {"O": p.O, "C": p.C, "E": p.E, "A": p.A, "N": p.N},
        "initial_memory_count": len(mems),
        # Seed memories so the UI shows the backstory the user gave the character
        # (these load on the backend outside the event bus, so they'd otherwise
        # be invisible until the first turn).
        "initial_memories": [_serialize_memory(m) for m in mems],
        "key_memory_ids": key_ids,        # promoted top memories (by id)
        "initial_facts": facts,           # seeded Prolog facts
        "config": {
            "retrieval_threshold": engram_config.RETRIEVAL_THRESHOLD,
            "top_k": engram_config.TOP_K_RETRIEVAL,
            "session_window": engram_config.SESSION_WINDOW,
            "evict_batch": engram_config.EVICT_BATCH,
            "key_memory_percentile": engram_config.KEY_MEMORY_PERCENTILE,
            "decay_rate": engram_config.DECAY_RATE,
        },
    }


# ---------------------------------------------------------------------------
# Onboarding-wizard LLM helpers (shared by /infer_character, /infer_ocean,
# /appearance so the legacy endpoints stay behaviour-identical)
# ---------------------------------------------------------------------------

def _rate_gate(byok: Optional[str], request: Request) -> Optional[JSONResponse]:
    """Apply the shared-key rate limit when no BYOK key is supplied.

    Returns a 429 JSONResponse to short-circuit with, or None to proceed.
    """
    if byok:
        return None
    ip = _client_ip(request)
    ok, retry = _check_rate(ip)
    if not ok:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "retry_after_s": retry},
        )
    return None


def _format_qa(qa: Optional[list]) -> str:
    """Render onboarding Q&A pairs into a prompt block."""
    qa_lines = []
    for item in (qa or []):
        if isinstance(item, dict):
            q = str(item.get("question", "")).strip()
            a = str(item.get("answer", "")).strip()
            if q or a:
                qa_lines.append(f"Q: {q}\nA: {a}")
    return "\n\n".join(qa_lines) or "(no answers provided)"


# Shared between /appearance and /infer_character so the appearance rules
# stay identical (Meshy needs the exact T-pose suffix).
_APPEARANCE_RULES = (
    "- Focus on body build, posture, facial structure, and expression.\n"
    "- Let the personality show through body language.\n"
    "- Do NOT mention any colors.\n"
    "- End the description with exactly: T-pose, humanoid, game character, "
    "realistic proportions."
)

_OCEAN_TRAIT_LINES = (
    "- O: Openness\n- C: Conscientiousness\n- E: Extraversion\n"
    "- A: Agreeableness\n- N: Neuroticism"
)

_ARCHETYPE_RULE = (
    "a short, evocative archetype: a 2-4 word "
    "title that captures their personality like a character class or trope "
    "(e.g. \"The Wary Sentinel\", \"The Open Wanderer\", \"The Rigid "
    "Archivist\", \"The Warm Broker\")"
)


def _parse_ocean_result(result: dict) -> tuple[dict[str, float], str, str]:
    """Pull (ocean, summary, archetype) out of an LLM JSON result, clamped."""
    ocean: dict[str, float] = {}
    for trait in ('O', 'C', 'E', 'A', 'N'):
        try:
            val = float(result.get(trait, 0.5))
        except (TypeError, ValueError):
            val = 0.5
        ocean[trait] = round(max(0.0, min(1.0, val)), 3)

    summary = result.get("summary", "")
    if not isinstance(summary, str):
        summary = str(summary)

    archetype = result.get("archetype", "")
    if not isinstance(archetype, str):
        archetype = str(archetype)
    archetype = archetype.strip() or "The Enigma"

    return ocean, summary, archetype


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

api = FastAPI(title="Engram Live Demo", version="0.1.0")

api.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://.*\.github\.io|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@api.get("/health")
async def health() -> dict:
    """Liveness + config probe."""
    _purge_expired()
    return {
        "ok": True,
        "presets": list(PRESETS.keys()),
        "shared_key_configured": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
    }


@api.post("/start")
async def start(body: StartReq, request: Request, x_anthropic_key: Optional[str] = Header(None)) -> dict:
    """Bootstrap a session: build the agent, capture the header, return session_id."""
    _purge_expired()

    if not body.custom and body.npc_id not in PRESETS:
        raise HTTPException(400, f"unknown npc_id; options: {list(PRESETS.keys())}")

    npc_id = _slugify(body.name) if body.custom else body.npc_id

    byok = body.anthropic_key or body.gemini_key or x_anthropic_key
    if not byok:
        ip = _client_ip(request)
        ok, retry = _check_rate(ip)
        if not ok:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "retry_after_s": retry},
            )

    llm = _resolve_llm(byok)
    session_id = str(uuid.uuid4())
    data_dir_root = _make_data_dir(session_id, npc_id)

    # Restore saved device memory before agent construction so NPCAgent
    # finds state.json and skips _init_backstory().
    speaker_line = None
    if body.device_id:
        restored = _restore_device_memory(body.device_id, npc_id, data_dir_root)
        log.info("device %s npc=%s restored=%s", body.device_id[:8], npc_id, restored)
        # Speaker awareness: the NPC knows different people talk to it across
        # visits. Situational context only — personality still flows from OCEAN.
        try:
            ctx = engram_db.speaker_context(npc_id, body.device_id)
            who = ctx["speaker_label"] + (
                ", who has visited you before" if ctx["is_returning"]
                else ", meeting you for the first time"
            )
            n_speakers = ctx["distinct_speakers"]
            speaker_line = (
                "Situational context: different people speak with you across "
                f"visits, and you can tell them apart. Right now you are "
                f"speaking with {who}. You have met "
                f"{n_speakers} different {'person' if n_speakers == 1 else 'people'} "
                f"across {ctx['total_sessions'] + 1} visits."
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("speaker context failed npc=%s: %s", npc_id, exc)

    if body.custom:
        traits = _ocean_from_dict(body.ocean)
        profile = OCEANProfile(name=body.name or npc_id, **traits)
        config = NPCConfig(
            npc_id=npc_id,
            name=body.name or "Custom NPC",
            persona=body.persona or "A custom character.",
            backstory=body.backstory or [],
            profile=profile,
            initial_facts=body.facts or [],
        )
    else:
        config = get_preset(body.npc_id)
        if body.ocean:
            for trait in ('O', 'C', 'E', 'A', 'N'):
                if trait in body.ocean:
                    try:
                        val = float(body.ocean[trait])
                    except (TypeError, ValueError):
                        raise HTTPException(400, f"ocean.{trait} must be a number")
                    if not 0.0 <= val <= 1.0:
                        raise HTTPException(400, f"ocean.{trait} must be in [0, 1]")
                    setattr(config.profile, trait, round(val, 3))

    if speaker_line:
        config.persona = f"{config.persona}\n\n{speaker_line}"

    # Run agent construction OUTSIDE the live event bus, backstory init can
    # emit dozens of memory_added events that would otherwise race the SSE
    # response. We surface them in the header instead.
    try:
        agent = NPCAgent(config, llm, data_dir=data_dir_root)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(data_dir_root, ignore_errors=True)
        log.exception("NPCAgent init failed for npc=%s", npc_id)
        raise HTTPException(503, f"agent init failed: {exc}")

    header = _build_header(agent)
    SESSIONS[session_id] = _Session(
        npc_id=npc_id, agent=agent, llm=llm, data_dir=data_dir_root,
        device_id=body.device_id,
    )
    try:
        engram_db.record_session_start(
            session_id, npc_id, body.device_id,
            npc_name=config.name, kind="custom" if body.custom else "preset",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("session start record failed: %s", exc)
    log.info("session start sid=%s npc=%s mem=%d",
             session_id, npc_id, header["initial_memory_count"])
    return {"session_id": session_id, "header": header}


@api.post("/turn")
async def turn(body: TurnReq, request: Request, x_anthropic_key: Optional[str] = Header(None)):
    """Run one pipeline turn and stream every emitted event back as SSE."""
    _purge_expired()

    sess = SESSIONS.get(body.session_id)
    if sess is None:
        raise HTTPException(404, "session_not_found")
    if sess.turn_count >= HARD_TURN_CAP:
        raise HTTPException(410, "session_cap_reached")

    byok = body.anthropic_key or body.gemini_key or x_anthropic_key
    if not byok:
        ip = _client_ip(request)
        ok, retry = _check_rate(ip)
        if not ok:
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "retry_after_s": retry},
            )

    sess.last_used_ts = time.time()

    # BYOK can hot-swap the LLM mid-session (visitor pastes a key).
    if byok:
        try:
            new_llm = GeminiClient(api_key=byok)
            sess.llm = new_llm
            sess.agent.llm = new_llm
            sess.agent.memory_manager.llm_client = new_llm
        except Exception as exc:  # noqa: BLE001
            log.warning("BYOK swap failed sid=%s: %s", body.session_id, exc)
            raise HTTPException(503, f"BYOK init failed: {exc}")

    queue: asyncio.Queue = asyncio.Queue()
    SENTINEL: object = object()
    loop = asyncio.get_running_loop()

    def on_event(event: dict) -> None:
        # Called from the executor thread (sync emit), must hop back to the loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def event_stream():
        bus.activate()
        unsubscribe = bus.subscribe(on_event)
        try:
            turn_task = loop.run_in_executor(None, sess.agent.run_turn, body.player_input)

            async def watch_turn() -> None:
                try:
                    await turn_task
                    _save_device_memory(sess)
                except Exception as exc:  # noqa: BLE001
                    log.exception("turn failed sid=%s", body.session_id)
                    queue.put_nowait({
                        "t": -1,
                        "type": "error",
                        "payload": {"message": str(exc)[:200]},
                    })
                finally:
                    queue.put_nowait(SENTINEL)

            asyncio.create_task(watch_turn())

            while True:
                event = await queue.get()
                if event is SENTINEL:
                    break
                yield {"data": json.dumps(event, ensure_ascii=False)}
        finally:
            unsubscribe()
            bus.deactivate()
            sess.turn_count += 1
            sess.last_used_ts = time.time()

    return EventSourceResponse(event_stream())


@api.post("/end")
async def end(request: Request) -> Response:
    """Drop a session and clean up its sandbox. Idempotent.

    Parses the body manually so it accepts both application/json and
    text/plain: the browser-close beacon must send text/plain, because
    cross-origin beacons cannot preflight an application/json payload.
    """
    _purge_expired()
    try:
        payload = json.loads((await request.body()).decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001
        payload = {}
    session_id = str(payload.get("session_id") or "")
    sess = SESSIONS.pop(session_id, None)
    if sess is not None:
        try:
            # Best-effort end_session for symmetry with chat.py, never let
            # cleanup throw across the network boundary.
            sess.agent.end_session()
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.end_session failed sid=%s: %s", session_id, exc)
        _save_device_memory(sess)
        try:
            engram_db.record_session_end(session_id, sess.turn_count)
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(sess.data_dir, ignore_errors=True)
        log.info("session end sid=%s npc=%s turns=%d",
                 session_id, sess.npc_id, sess.turn_count)
    return Response(status_code=204)


@api.post("/waitlist")
async def waitlist(request: Request):
    """Store a waitlist signup {email, note}. Accepts JSON or text/plain.
    Returns ok even for duplicate emails (no signal leak, friendlier UX)."""
    ip = _client_ip(request)
    ok, retry = _check_rate(ip)
    if not ok:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "retry_after_s": retry},
        )
    try:
        payload = json.loads((await request.body()).decode("utf-8") or "{}")
    except Exception:  # noqa: BLE001
        payload = {}
    email = str(payload.get("email") or "").strip()
    note = str(payload.get("note") or "").strip()[:2000]
    if "@" not in email or "." not in email.rpartition("@")[2] or len(email) > 254:
        raise HTTPException(400, "valid email required")
    try:
        engram_db.add_waitlist(email, note)
    except Exception as exc:  # noqa: BLE001
        log.warning("waitlist add failed: %s", exc)
        raise HTTPException(503, "waitlist storage unavailable")
    log.info("waitlist signup %s", email)
    return {"ok": True}


@api.post("/infer_character")
def infer_character(body: InferCharacterReq, request: Request, x_anthropic_key: Optional[str] = Header(None)) -> dict:
    """Combined onboarding inference: OCEAN + summary + archetype + 3D
    appearance description in ONE Anthropic call.

    Replaces the wizard's /infer_ocean + /appearance pair (both kept for
    backwards compat). The prompt is schema-shaped to mirror the JSON output.
    """
    _purge_expired()

    byok = body.anthropic_key or x_anthropic_key
    gate = _rate_gate(byok, request)
    if gate is not None:
        return gate

    llm = _resolve_llm(byok)

    qa_text = _format_qa(body.qa)
    name = (body.name or "").strip() or "the character"

    prompt = (
        "You are a personality psychologist and character designer. Below are "
        "question-and-answer pairs describing what kind of person a fictional "
        f"character named {name} is.\n\n"
        f"{qa_text}\n\n"
        "Do ALL of the following:\n"
        "1. Infer this character's Big Five (OCEAN) personality traits, each "
        "on a 0.0-1.0 scale (0.0 = very low, 1.0 = very high):\n"
        f"{_OCEAN_TRAIT_LINES}\n"
        "2. Write a 2-3 sentence personality summary of this character.\n"
        f"3. Give this character {_ARCHETYPE_RULE}.\n"
        "4. Write a 2-3 sentence physical appearance description for a 3D "
        "character model, consistent with the inferred personality:\n"
        f"{_APPEARANCE_RULES}\n\n"
        "Respond with JSON only, exactly this shape:\n"
        "{\n"
        '  "O": <float 0.0-1.0>,\n'
        '  "C": <float 0.0-1.0>,\n'
        '  "E": <float 0.0-1.0>,\n'
        '  "A": <float 0.0-1.0>,\n'
        '  "N": <float 0.0-1.0>,\n'
        '  "summary": "<2-3 sentence personality summary>",\n'
        '  "archetype": "<2-4 word archetype>",\n'
        '  "appearance_description": "<2-3 sentences ending with: T-pose, '
        'humanoid, game character, realistic proportions.>"\n'
        "}"
    )

    try:
        result = llm.generate_json(prompt, max_tokens=768)
    except Exception as exc:  # noqa: BLE001
        log.exception("infer_character generate_json failed")
        raise HTTPException(503, f"inference failed: {exc}")

    if not isinstance(result, dict):
        result = {}

    ocean, summary, archetype = _parse_ocean_result(result)

    appearance = result.get("appearance_description", "")
    if not isinstance(appearance, str):
        appearance = str(appearance)
    appearance = appearance.strip()

    log.info("infer_character ocean=%s archetype=%s appearance_len=%d",
             ocean, archetype, len(appearance))
    return {
        "ocean": ocean,
        "summary": summary,
        "archetype": archetype,
        "appearance_description": appearance,
    }


@api.post("/infer_ocean")
def infer_ocean(body: InferOceanReq, request: Request, x_anthropic_key: Optional[str] = Header(None)) -> dict:
    """Infer Big Five OCEAN scores + a personality summary from Q&A pairs.

    Legacy endpoint, the wizard now calls /infer_character (one combined
    LLM call). Kept for backwards compat; shares helpers with it.
    """
    _purge_expired()

    byok = body.anthropic_key or x_anthropic_key
    gate = _rate_gate(byok, request)
    if gate is not None:
        return gate

    llm = _resolve_llm(byok)

    qa_text = _format_qa(body.qa)

    prompt = (
        "You are a personality psychologist. Below are question-and-answer pairs "
        "describing what kind of person a fictional character is.\n\n"
        f"{qa_text}\n\n"
        "Infer this character's Big Five (OCEAN) personality traits, each on a "
        "0.0-1.0 scale (0.0 = very low, 1.0 = very high):\n"
        f"{_OCEAN_TRAIT_LINES}\n\n"
        "Also write a 2-3 sentence personality summary of this character.\n\n"
        f"Finally, give this character {_ARCHETYPE_RULE}.\n\n"
        "Respond with JSON only, with keys O, C, E, A, N (floats 0.0-1.0), "
        "summary (string), and archetype (string)."
    )

    try:
        result = llm.generate_json(prompt)
    except Exception as exc:  # noqa: BLE001
        log.exception("infer_ocean generate_json failed")
        raise HTTPException(503, f"inference failed: {exc}")

    if not isinstance(result, dict):
        result = {}

    ocean, summary, archetype = _parse_ocean_result(result)

    log.info("infer_ocean ocean=%s archetype=%s", ocean, archetype)
    return {"ocean": ocean, "summary": summary, "archetype": archetype}


@api.post("/appearance")
def appearance(body: AppearanceReq, request: Request, x_anthropic_key: Optional[str] = Header(None)) -> dict:
    """Generate a 3D character appearance description from name, persona, OCEAN.

    The wizard only calls this on "Regenerate"; the first description comes
    from the combined /infer_character call. Shares _APPEARANCE_RULES with it.
    """
    _purge_expired()

    byok = body.anthropic_key or x_anthropic_key
    gate = _rate_gate(byok, request)
    if gate is not None:
        return gate

    llm = _resolve_llm(byok)

    ocean = _ocean_from_dict(body.ocean)
    ocean_text = ", ".join(f"{t}={ocean[t]}" for t in ('O', 'C', 'E', 'A', 'N'))

    prompt = (
        "Write a 2-3 sentence physical appearance description for a 3D character "
        "model.\n\n"
        f"Name: {body.name}\n"
        f"Persona: {body.persona or '(none)'}\n"
        f"OCEAN personality (0.0-1.0): {ocean_text}\n\n"
        "Rules:\n"
        f"{_APPEARANCE_RULES}\n\n"
        "Respond with the description text only."
    )

    try:
        description = llm.generate(prompt, max_tokens=220)
    except Exception as exc:  # noqa: BLE001
        log.exception("appearance generate failed")
        raise HTTPException(503, f"appearance generation failed: {exc}")

    description = (description or "").strip()
    log.info("appearance generated name=%s len=%d", body.name, len(description))
    return {"description": description}


@api.post("/greeting")
def greeting(body: GreetingReq, request: Request, x_anthropic_key: Optional[str] = Header(None)) -> dict:
    """Generate a character's first spoken line to a stranger, in character."""
    _purge_expired()

    byok = body.anthropic_key or x_anthropic_key
    gate = _rate_gate(byok, request)
    if gate is not None:
        return gate

    llm = _resolve_llm(byok)

    ocean = _ocean_from_dict(body.ocean)
    ocean_text = ", ".join(f"{t}={ocean[t]}" for t in ('O', 'C', 'E', 'A', 'N'))

    prompt = (
        "You are roleplaying as a fictional character meeting a stranger for the "
        "first time. Write the VERY FIRST thing this character says out loud.\n\n"
        f"Name: {body.name}\n"
        f"Persona: {body.persona or '(none)'}\n"
        f"OCEAN personality (0.0-1.0): {ocean_text}\n\n"
        "Let the personality color the greeting:\n"
        "- High Neuroticism (N) = guarded, wary, anxious opening; low N = relaxed, secure.\n"
        "- High Extraversion (E) = warm, forward, talkative; low E = reserved, brief.\n"
        "- Low Agreeableness (A) = curt, blunt, suspicious; high A = friendly, welcoming.\n"
        "- High Openness (O) = curious, imaginative phrasing; low O = plain, conventional.\n"
        "- High Conscientiousness (C) = careful, orderly; low C = casual, loose.\n\n"
        "Rules:\n"
        "- Stay fully in character, shaped by their name, persona, and personality.\n"
        "- A natural greeting or opening line, 1-2 sentences.\n"
        "- No narration, no stage directions, no quotation marks.\n"
        "- Output only what the character says, nothing else."
    )

    try:
        greeting_text = llm.generate(prompt, max_tokens=120)
    except Exception as exc:  # noqa: BLE001
        log.exception("greeting generate failed")
        raise HTTPException(503, f"greeting generation failed: {exc}")

    greeting_text = (greeting_text or "").strip().strip('"').strip("'").strip()
    if body.session_id and greeting_text:
        sess = SESSIONS.get(body.session_id)
        if sess is not None:
            # Record the unprompted opener so later turns know it was said.
            sess.agent.history.append(
                {"player": "(approaches for the first time)", "npc": greeting_text}
            )
            sess.last_used_ts = time.time()
    log.info("greeting generated name=%s len=%d", body.name, len(greeting_text))
    return {"greeting": greeting_text}


@api.post("/generate_character")
async def generate_character(body: GenerateCharacterReq, request: Request):
    """Kick off a background Meshy text-to-3D pipeline (preview->refine->rig).

    Returns a job_id the frontend polls via /character_status. If no Meshy key
    is configured, returns {"job_id": None, "disabled": True} so the frontend
    keeps its grey placeholder instead of erroring.
    """
    # Meshy jobs burn paid credits; always gate on the per-IP limiter.
    ok, retry = _check_rate(_client_ip(request))
    if not ok:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "retry_after_s": retry},
        )
    if not MESHY_API_KEY:
        log.info("generate_character disabled (no MESHY_API_KEY)")
        return {"job_id": None, "disabled": True}

    job_id = uuid.uuid4().hex
    with _CHARACTER_JOBS_LOCK:
        CHARACTER_JOBS[job_id] = {
            "status": "running",
            "stage": "queued",
            "glb_url": None,
            "progress": 0,
            "error": None,
        }

    thread = threading.Thread(
        target=_generate_character_worker,
        args=(job_id, body.name, body.description),
        daemon=True,
    )
    thread.start()
    log.info("generate_character job=%s name=%s", job_id, body.name)
    return {"job_id": job_id}


@api.get("/character_status/{job_id}")
async def character_status(job_id: str) -> dict:
    """Return the current state of a character-generation job.

    Returns a status=error dict (not an exception) for unknown jobs so the
    frontend can stop polling cleanly.
    """
    job = _get_job(job_id)
    if job is None:
        return {"status": "error", "error": "unknown job"}
    return job


@api.get("/proxy_glb")
def proxy_glb(url: str, request: Request):
    """Stream a Meshy CDN GLB through our origin so the browser can load it
    without cross-origin (CORS) issues. Only allows Meshy asset URLs."""
    ok, retry = _check_rate(_client_ip(request))
    if not ok:
        return JSONResponse(
            status_code=429,
            content={"error": "rate_limited", "retry_after_s": retry},
        )
    if not (url.startswith("https://assets.meshy.ai/") or url.startswith("https://api.meshy.ai/")):
        raise HTTPException(400, "only meshy asset urls are allowed")
    try:
        r = requests.get(url, timeout=120, stream=True)
        r.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"fetch failed: {exc}")
    return Response(
        content=r.content,
        media_type="model/gltf-binary",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ---------------------------------------------------------------------------
# Modal wrapping, optional. The block must remain importable even when Modal
# is not installed (the local dev path uses uvicorn, no Modal needed).
# ---------------------------------------------------------------------------

try:
    import modal  # type: ignore
except ImportError:  # pragma: no cover, local-dev fallback
    modal = None  # type: ignore


if modal is not None:
    # Legacy pre-SQLite store: kept read-only so old device memories migrate
    # lazily into the DB on first lookup (see _device_get).
    try:
        _modal_device_dict = modal.Dict.from_name("engram-device-memory", create_if_missing=True)

        def _legacy_device_get(key: str) -> dict | None:  # noqa: F811
            try:
                return _modal_device_dict.get(key)
            except Exception:
                return None

    except Exception as _exc:
        log.warning("legacy modal.Dict unavailable, migration disabled: %s", _exc)

    app = modal.App("engram-demo")
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("swi-prolog")
        .pip_install_from_requirements("backend/requirements.txt")
        .add_local_dir(
            ".",
            remote_path="/root/engram",
            ignore=[".venv", ".git", "node_modules", "docs/sessions"],
        )
    )

    # Persistent SQLite home; db.py defaults to /data/engram.sqlite3.
    _db_volume = modal.Volume.from_name("engram-db", create_if_missing=True)

    @app.function(
        image=image,
        # Create with:
        #   modal secret create engramchar-keys \
        #     ANTHROPIC_API_KEY=sk-ant-... VOYAGE_API_KEY=pa-... MESHY_API_KEY=msy_...
        secrets=[modal.Secret.from_name("engramchar-keys")],
        volumes={"/data": _db_volume},
        # Pinned to a single container so SESSIONS (in-process dict) and the
        # bus singleton + Prolog state all stay coherent across requests.
        # Sticky sessions across containers would require modal.Dict + agent
        # serialization (out of scope for the demo).
        min_containers=1,
        max_containers=1,
        scaledown_window=600,
        timeout=120,
    )
    @modal.concurrent(max_inputs=1)  # serialize: bus + SESSIONS are process-global
    @modal.asgi_app()
    def fastapi_app():
        # Make sure /root/engram/src is importable inside the container.
        modal_src = "/root/engram/src"
        if modal_src not in sys.path:
            sys.path.insert(0, modal_src)
        # Flush every committed DB write to the volume so memory survives
        # container restarts and redeploys.
        try:
            engram_db.on_write = _db_volume.commit
            engram_db.init_db()
        except Exception as _db_exc:  # noqa: BLE001
            log.warning("DB volume hookup failed: %s", _db_exc)
        return api
