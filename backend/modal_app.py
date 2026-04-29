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
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse


# ---------------------------------------------------------------------------
# Path setup — make sure ``engram`` is importable in both Modal and local dev
# ---------------------------------------------------------------------------

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_THIS_DIR)

# Local dev: src is at <repo>/src. Modal: image copies the repo to /root/engram
# and we add /root/engram/src in the modal entrypoint below.
_LOCAL_SRC = os.path.join(REPO_ROOT, "src")
if os.path.isdir(_LOCAL_SRC) and _LOCAL_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SRC)

from engram import config as engram_config  # noqa: E402
from engram.llm.client import GeminiClient  # noqa: E402
from engram.npc import NPCAgent  # noqa: E402
from engram.observability import bus  # noqa: E402
from engram.presets import PRESETS, get_preset  # noqa: E402


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
RATE_LIMIT_PER_MIN = 5
RATE_LIMIT_PER_DAY = 50

_SESSION_BASE_DIR = "/tmp/engram_sessions"


# ---------------------------------------------------------------------------
# In-process state
# ---------------------------------------------------------------------------

@dataclass
class _Session:
    npc_id: str
    agent: NPCAgent
    llm: GeminiClient
    data_dir: str
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
    gemini_key: Optional[str] = None


class TurnReq(BaseModel):
    session_id: str
    player_input: str
    gemini_key: Optional[str] = None


class EndReq(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return (request.client.host if request.client else "") or "unknown"


def _purge_expired() -> None:
    """Drop sessions older than the TTL. Runs at the top of every request."""
    now = time.time()
    expired = [sid for sid, s in SESSIONS.items() if now - s.last_used_ts > SESSION_TTL_S]
    for sid in expired:
        s = SESSIONS.pop(sid, None)
        if s is not None:
            shutil.rmtree(s.data_dir, ignore_errors=True)
            log.info("session expired sid=%s npc=%s", sid, s.npc_id)


def _check_rate(ip: str) -> tuple[bool, int]:
    """Sliding-window per-IP rate limit. Returns (ok, retry_after_seconds)."""
    now = time.time()
    minute_ago = now - 60.0
    day_ago = now - 86400.0

    bucket = _RATE.get(ip, [])
    # Inline cleanup — drop anything older than the daily window.
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
    key = (byok or os.environ.get("GEMINI_API_KEY", "")).strip()
    if not key:
        raise HTTPException(503, "No Gemini API key configured and none provided")
    try:
        return GeminiClient(api_key=key)
    except Exception as exc:  # noqa: BLE001
        log.warning("GeminiClient init failed: %s", exc)
        raise HTTPException(503, f"Gemini client init failed: {exc}")


def _make_data_dir(session_id: str, npc_id: str) -> str:
    """Per-session sandbox under /tmp. Reuses pre-baked preset data when present."""
    os.makedirs(_SESSION_BASE_DIR, exist_ok=True)
    dst = os.path.join(_SESSION_BASE_DIR, session_id)
    src = os.path.join(REPO_ROOT, "data", npc_id)
    if os.path.isdir(src):
        # copytree won't overwrite existing dirs without dirs_exist_ok=True (Py 3.8+).
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        os.makedirs(dst, exist_ok=True)
    return dst


def _build_header(agent: NPCAgent) -> dict:
    """Mirror chat.py's _build_viz_header but populate from a live agent."""
    p = agent.config.profile
    return {
        "npc_id": agent.config.npc_id,
        "npc_name": agent.config.name,
        "persona": agent.config.persona,
        "baseline_ocean": {"O": p.O, "C": p.C, "E": p.E, "A": p.A, "N": p.N},
        "initial_memory_count": len(agent.memory_manager.all_memories),
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
        "shared_key_configured": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
    }


@api.post("/start")
async def start(body: StartReq, request: Request, x_gemini_key: Optional[str] = Header(None)) -> dict:
    """Bootstrap a session: build the agent, capture the header, return session_id."""
    _purge_expired()

    if body.npc_id not in PRESETS:
        raise HTTPException(400, f"unknown npc_id; options: {list(PRESETS.keys())}")

    byok = body.gemini_key or x_gemini_key
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
    data_dir_root = _make_data_dir(session_id, body.npc_id)

    config = get_preset(body.npc_id)

    # Run agent construction OUTSIDE the live event bus — backstory init can
    # emit dozens of memory_added events that would otherwise race the SSE
    # response. We surface them in the header instead.
    try:
        agent = NPCAgent(config, llm, data_dir=data_dir_root)
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(data_dir_root, ignore_errors=True)
        log.exception("NPCAgent init failed for npc=%s", body.npc_id)
        raise HTTPException(503, f"agent init failed: {exc}")

    header = _build_header(agent)
    SESSIONS[session_id] = _Session(
        npc_id=body.npc_id, agent=agent, llm=llm, data_dir=data_dir_root,
    )
    log.info("session start sid=%s npc=%s mem=%d",
             session_id, body.npc_id, header["initial_memory_count"])
    return {"session_id": session_id, "header": header}


@api.post("/turn")
async def turn(body: TurnReq, request: Request, x_gemini_key: Optional[str] = Header(None)):
    """Run one pipeline turn and stream every emitted event back as SSE."""
    _purge_expired()

    sess = SESSIONS.get(body.session_id)
    if sess is None:
        raise HTTPException(404, "session_not_found")
    if sess.turn_count >= HARD_TURN_CAP:
        raise HTTPException(410, "session_cap_reached")

    byok = body.gemini_key or x_gemini_key
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
        # Called from the executor thread (sync emit) — must hop back to the loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def event_stream():
        bus.activate()
        unsubscribe = bus.subscribe(on_event)
        try:
            turn_task = loop.run_in_executor(None, sess.agent.run_turn, body.player_input)

            async def watch_turn() -> None:
                try:
                    await turn_task
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
async def end(body: EndReq) -> Response:
    """Drop a session and clean up its sandbox. Idempotent."""
    _purge_expired()
    sess = SESSIONS.pop(body.session_id, None)
    if sess is not None:
        try:
            # Best-effort end_session for symmetry with chat.py — never let
            # cleanup throw across the network boundary.
            sess.agent.end_session()
        except Exception as exc:  # noqa: BLE001
            log.warning("agent.end_session failed sid=%s: %s", body.session_id, exc)
        shutil.rmtree(sess.data_dir, ignore_errors=True)
        log.info("session end sid=%s npc=%s turns=%d",
                 body.session_id, sess.npc_id, sess.turn_count)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Modal wrapping — optional. The block must remain importable even when Modal
# is not installed (the local dev path uses uvicorn, no Modal needed).
# ---------------------------------------------------------------------------

try:
    import modal  # type: ignore
except ImportError:  # pragma: no cover — local-dev fallback
    modal = None  # type: ignore


if modal is not None:
    stub = modal.App("engram-demo")
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

    @stub.function(
        image=image,
        secrets=[modal.Secret.from_name("engram-gemini-key")],
        min_containers=0,
        max_containers=4,
        scaledown_window=120,
        timeout=120,
    )
    @modal.concurrent(max_inputs=1)  # serialize: bus + SESSIONS are process-global
    @modal.asgi_app()
    def fastapi_app():
        # Make sure /root/engram/src is importable inside the container.
        modal_src = "/root/engram/src"
        if modal_src not in sys.path:
            sys.path.insert(0, modal_src)
        return api
