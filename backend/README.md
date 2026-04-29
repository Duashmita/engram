# Engram live-demo backend

A FastAPI app that runs the Engram pipeline once per visitor and streams
pipeline events back as Server-Sent Events. Wrapped as a Modal ASGI app so
that a single command ships it to production. Locally runnable as plain
FastAPI via `uvicorn` for development without Modal.

## Local development

```bash
# 1. system deps (Prolog is required by the pipeline)
brew install swi-prolog       # or: apt install swi-prolog

# 2. python deps
pip install -r backend/requirements.txt

# 3. Gemini key
export GEMINI_API_KEY=...     # or put it in a .env at the repo root

# 4. run
uvicorn backend.modal_app:api --port 8000 --reload
```

Smoke test:

```bash
curl http://localhost:8000/health
# {"ok": true, "presets": ["jeanie", ...], "shared_key_configured": true}
```

## Modal deploy

```bash
pip install modal
modal token new                       # one-time auth

# Store the shared Gemini key as a Modal Secret (named env var).
modal secret create engram-gemini-key GEMINI_API_KEY=...

# Ship it.
modal deploy backend/modal_app.py
```

Modal prints a public URL when it's done. Copy that URL into
`docs/config.js` as `BACKEND_URL` so the GH Pages frontend points at it.

For iteration, `modal serve backend/modal_app.py` gives you a live-reload
URL backed by your laptop.

## API surface

| method | path     | purpose                                              |
|--------|----------|------------------------------------------------------|
| GET    | /health  | liveness + preset list + whether shared key exists   |
| POST   | /start   | bootstrap a session, return `{session_id, header}`   |
| POST   | /turn    | run one pipeline turn, stream events as SSE          |
| POST   | /end     | drop a session and clean up its sandbox (idempotent) |

`/turn` is the interesting one — every `bus.emit(...)` from the pipeline
becomes a `data:` line in the SSE stream, in real time. The header you got
from `/start` is *not* re-sent on `/turn`; it's delivered once per session
out of band.

## Operational notes

- **Live mode is ephemeral.** It does NOT update `docs/sessions/manifest.json`.
  Replay mode keeps using whatever NDJSON files are checked into
  `docs/sessions/`.
- **Per-session sandbox.** Each session gets a fresh `/tmp/engram_sessions/<sid>/`
  copied from `data/<npc_id>/` if a pre-baked one exists (saves the ~30
  embed calls you'd otherwise pay on backstory init).
- **Hard caps.** 30 turns per session, 1 hour TTL.
- **Rate limits.** 5 starts/turns per minute and 50 per day per IP. Visitors
  who paste their own Gemini key in the UI's settings popover bypass the
  limits (BYOK path).
- **Single-tenant container.** `modal.concurrent(max_inputs=1)` serializes
  requests per container — the observability bus is a process-global
  singleton, so we don't try to interleave two pipelines through one
  process. Modal scales out by adding more containers, not threads.

## Files

- `modal_app.py` — the FastAPI app and the Modal wrapping.
- `requirements.txt` — pinned deps.
