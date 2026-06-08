# Deploying engram-chars

Two pieces go live independently:
- Backend (the pipeline + LLM/Meshy endpoints) on Modal
- Frontend (the static site) on GitHub Pages

## 1. Backend on Modal

```bash
pip install modal
modal token new                      # one-time auth

# Store the three API keys as one Modal secret named "engram-keys"
modal secret create engram-keys \
  ANTHROPIC_API_KEY=sk-ant-...\
  VOYAGE_API_KEY=pa-...\
  MESHY_API_KEY=msy_...

# Deploy
modal deploy backend/modal_app.py
```

Modal prints a URL like `https://<you>--engram-demo-fastapi-app.modal.run`.
Copy it.

## 2. Point the frontend at the deployed backend

Edit `docs/config.js` and set `MODAL_BACKEND_URL` to the URL Modal printed.
The file auto-selects: localhost uses `http://localhost:8000`, everything else
(GitHub Pages) uses `MODAL_BACKEND_URL`.

## 3. Frontend on GitHub Pages

GitHub Pages serves either the repo root or `/docs` at the repo root. Since this
project keeps the site under `engram-chars/docs`, publish the site contents to a
dedicated `gh-pages` branch (already prepared by the deploy script below):

```bash
# from the engram repo root, with the engram-chars-3d branch checked out
bash engram-chars/scripts/publish_pages.sh
```

Then in the GitHub repo: Settings, Pages, Source = `gh-pages` branch, `/ (root)`.
The site goes live at `https://<user>.github.io/<repo>/`.

CORS is already configured on the backend to allow `*.github.io`.

## Notes
- Preset characters ship as preloaded GLBs. Only `base.glb` is needed for the
  live site (animations are off by default), so the Pages payload stays small.
- Custom characters generate their 3D model in the background via Meshy and the
  model is loaded through the backend `/proxy_glb` endpoint to avoid CORS.
