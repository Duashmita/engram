// docs/config.js
// The backend URL is chosen automatically:
//   - on localhost it points at your local uvicorn (port 8000)
//   - anywhere else (e.g. GitHub Pages) it uses the deployed Modal URL below
// After running `modal deploy backend/modal_app.py`, paste the printed URL into
// MODAL_BACKEND_URL.

const MODAL_BACKEND_URL = "https://duashmita--engram-demo-fastapi-app.modal.run";

const _host = location.hostname;
const _isLocal = _host === "localhost" || _host === "127.0.0.1" || _host === "";

export const BACKEND_URL = _isLocal ? "http://localhost:8000" : MODAL_BACKEND_URL;
