# Deploying the Engram Live Demo

This guide walks through shipping the Engram demo end-to-end: a static
GitHub Pages frontend that talks to a Modal-hosted FastAPI backend running
the full Engram pipeline per visitor.

Audience: the project owner and reviewers reproducing the deploy. Tone:
factual; copy-pasteable shell commands.

---

## 1. What you're deploying

The static site under `docs/` is a pure-HTML/JS visualizer that, in replay
mode, plays back NDJSON event logs from `docs/sessions/` and animates the
OCEAN radar, pipeline timeline, and memory panels. The live backend at
`backend/modal_app.py` is a Modal-hosted FastAPI app that runs the Engram
pipeline (threat assessment, personality-weighted retrieval, Prolog
contradiction check, dialogue generation, consolidation) per visitor and
streams pipeline events over Server-Sent Events. The two are linked by
`docs/config.js`, which exposes a `BACKEND_URL` constant the frontend
reads at boot to decide where to open SSE streams and POST chat turns.

---

## 2. Prerequisites

- Python 3.11+ (Modal images use 3.11; local dev tested with 3.9+ but
  3.11 is preferred to match the deployed runtime).
- A Gemini API key. Get one at https://aistudio.google.com/apikey.
- A Modal account; free tier is sufficient for a low-traffic demo. Sign
  up at https://modal.com.
- A GitHub repository with Pages enabled (Settings -> Pages -> Source =
  "GitHub Actions"; see step 4.3 below for the one-time toggle).
- SWI-Prolog installed locally if you plan to run the backend without
  Modal (`brew install swi-prolog` on macOS, `apt-get install
  swi-prolog` on Debian/Ubuntu).

---

## 3. Backend deploy (Modal)

1. Install the Modal CLI. Either globally:

   ```bash
   pip install modal
   ```

   or via the project venv:

   ```bash
   pip install -r backend/requirements.txt
   ```

2. Authorize the CLI. This opens a browser tab and writes credentials to
   `~/.modal.toml`:

   ```bash
   modal token new
   ```

3. Create the Gemini secret. The backend reads `GEMINI_API_KEY` from a
   Modal secret named `engram-gemini-key`:

   ```bash
   modal secret create engram-gemini-key GEMINI_API_KEY=<your_key>
   ```

4. (Optional but recommended) Pre-bake backstory artifacts locally
   before the first deploy so the first request to each preset doesn't
   pay the cold backstory-generation cost:

   ```bash
   GEMINI_API_KEY=<your_key> python3 backend/prebake.py
   ```

   This populates `data/<preset>/` for all six presets. The deploy step
   below bundles `data/` into the image.

5. Deploy the FastAPI app. Modal builds the image, pushes the artifact,
   and prints a stable URL:

   ```bash
   modal deploy backend/modal_app.py
   ```

   Output ends with a line like:

   ```
   https://<workspace>--engram-demo-fastapi-app.modal.run
   ```

   Save this URL; you'll wire it into the frontend in step 4.

6. Smoke-test the deploy. The included script hits `/health`,
   `/presets`, opens an SSE stream, and posts a turn:

   ```bash
   bash backend/smoke.sh https://<your-url>
   ```

   A successful run ends with `OK`. If any step fails, see section 8.

---

## 4. Frontend deploy (GitHub Pages)

1. Open `docs/config.js` and replace the localhost URL with the Modal
   URL from step 3.5:

   ```js
   window.ENGRAM_CONFIG = {
     BACKEND_URL: "https://<workspace>--engram-demo-fastapi-app.modal.run",
   };
   ```

2. Commit and push to `main`:

   ```bash
   git add docs/config.js
   git commit -m "wire frontend to deployed backend"
   git push origin main
   ```

   The included workflow `.github/workflows/pages.yml` runs on push and
   uploads `docs/` as the Pages artifact.

3. (One-time) Enable GitHub Actions as the Pages source. In the repo on
   GitHub: Settings -> Pages -> Source -> select **GitHub Actions**.
   Until this is set, the workflow will run but Pages won't serve the
   artifact.

4. Watch the run under the Actions tab. Within roughly one minute the
   site is live at:

   ```
   https://<your-username>.github.io/<repo>/
   ```

---

## 5. Verifying end-to-end

1. Open the GitHub Pages URL in a browser.
2. Toggle the mode switch to **Live**.
3. Pick **Jeanie** from the preset dropdown.
4. In the chat input, type `hi` and press Send.
5. Watch the panels animate as events stream in: the OCEAN radar should
   pulse, the pipeline timeline should fill stage by stage (threat ->
   retrieval -> contradiction -> generation -> consolidation), and the
   memory panel should show the retrieved memories with their scores.
   The transcript on the left fills in the assistant's reply.

A full turn typically completes in 3-6 seconds. The first turn after
the backend has been idle takes 5-8 seconds because of Modal cold-start.

---

## 6. Local-only dev (no Modal needed)

For iterating on the backend or frontend without redeploying:

1. Install backend dependencies and SWI-Prolog (system package):

   ```bash
   pip install -r backend/requirements.txt
   # macOS
   brew install swi-prolog
   # Debian/Ubuntu
   sudo apt-get install swi-prolog
   ```

2. Export your Gemini key in the shell:

   ```bash
   export GEMINI_API_KEY=...
   ```

3. Run the FastAPI app via uvicorn with autoreload:

   ```bash
   uvicorn backend.modal_app:api --port 8000 --reload
   ```

4. Serve the frontend. Either open `docs/index.html` directly in a
   browser, or (preferred, to avoid file:// CORS quirks) run a static
   server on a different port:

   ```bash
   python -m http.server 8001 -d docs
   ```

   Then visit http://localhost:8001/.

5. Confirm `docs/config.js` points at `http://localhost:8000`. That's
   the default committed value, so unless you've already wired it to a
   deployed backend you should be fine.

---

## 7. Cost & rate limits

- Gemini API cost per turn is roughly $0.001 to $0.01, depending on
  which model the pipeline routes to and how long the dialogue context
  has grown. The Modal secret `engram-gemini-key` pays for every visitor
  who doesn't bring their own key.
- Per-IP rate limit on the backend: 5 turns/minute and 50 turns/day.
  Enforced server-side; visitors hitting the cap get a 429.
- Per-session hard cap: 30 turns. After that the session is closed and
  the visitor must start a new one.
- Power users and paper reviewers can bypass the shared-key path and
  rate limits by clicking the **Settings** button in the UI and pasting
  their own Gemini API key. The key is held in browser memory only and
  forwarded per-request.
- Recommendation: set a Modal spending alarm at $10/month as a fail-safe
  in case rate limits are misconfigured or a key leaks. Modal dashboard
  -> Settings -> Billing -> spending alerts.

---

## 8. Troubleshooting

- **Cold-start latency.** First request after the Modal app has been
  idle takes 5-8 seconds (image pull + Python boot + Prolog init). The
  app ships with an optional warmup cron defined in `modal_app.py`; you
  can enable it for a smoother demo or accept the cold start for a
  low-traffic deploy.
- **CORS errors in the browser console.** The backend's allowed origins
  must include your GitHub Pages domain
  (`https://<your-username>.github.io`). If you renamed the repo or use
  a custom domain, edit the CORS middleware in `backend/modal_app.py`
  and redeploy with `modal deploy backend/modal_app.py`.
- **"Session not found" after a refresh.** Live sessions are ephemeral
  and pinned to a single backend container. After a refresh the
  session ID in localStorage no longer resolves. Click **Start session**
  in the UI to begin a new one; this is expected behavior, not a bug.
- **Replay mode shows nothing.** The replay UI reads
  `docs/sessions/manifest.json` for the list of available NDJSON files.
  If that manifest is empty or malformed, the dropdown stays empty and
  no session loads. Ship at least one NDJSON file (for example
  `sample-session.ndjson`) and keep it referenced from the manifest.
- **`modal deploy` fails with "secret not found".** You skipped step
  3.3. Run `modal secret create engram-gemini-key
  GEMINI_API_KEY=<your_key>` and redeploy.
- **Pages workflow runs but the site 404s.** You haven't switched the
  Pages source to "GitHub Actions" yet. See step 4.3.
