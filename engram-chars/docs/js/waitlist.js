// waitlist.js
// "Join waitlist" dialog: email + optional note, POSTed to the backend.
//
// Wiring: app.js should `import { initWaitlist } from './js/waitlist.js'` and
// call `initWaitlist()` once on startup (it is idempotent, calling it again is
// a no-op). The catalogue start screen lazy-imports `openWaitlist()` directly.
//
// Defensive by design: the `${BACKEND_URL}/waitlist` endpoint may not exist
// yet. On any failure (network error, non-2xx) we fall back to a mailto link
// so no signup is ever lost.

import { BACKEND_URL } from '../config.js';

const WAITLIST_EMAIL = 'asdua@ucsc.edu';

let _wired = false;

function $(id) {
  return document.getElementById(id);
}

function setStatus(msg, ok) {
  const status = $('waitlist-status');
  if (!status) return;
  status.textContent = msg || '';
  status.classList.toggle('ok', !!ok);
}

function mailtoFallback(email, note) {
  const body = 'Please add me to the Engram waitlist.\n\nEmail: ' + email
    + (note ? '\nWhat I am building: ' + note : '');
  const href = 'mailto:' + WAITLIST_EMAIL
    + '?subject=' + encodeURIComponent('Engram waitlist')
    + '&body=' + encodeURIComponent(body);
  try {
    window.location.href = href;
  } catch (_) { /* popup blockers etc., the status text still tells the user */ }
}

async function submit() {
  const emailInput = $('waitlist-email');
  const noteInput = $('waitlist-note');
  const sendBtn = $('waitlist-send');
  const email = (emailInput?.value || '').trim();
  const note = (noteInput?.value || '').trim();

  if (!email || !email.includes('@')) {
    setStatus('Please enter a valid email address.', false);
    emailInput?.focus();
    return;
  }

  if (sendBtn) { sendBtn.disabled = true; sendBtn.textContent = 'Joining…'; }
  setStatus('', false);

  let joined = false;
  try {
    const res = await fetch(`${BACKEND_URL}/waitlist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, note }),
    });
    joined = res.ok;
  } catch (_) {
    joined = false;
  }

  if (sendBtn) { sendBtn.disabled = false; sendBtn.textContent = 'Join waitlist'; }

  if (joined) {
    // Remember the signup so the in-chat auto-prompt never nags this user again.
    try { localStorage.setItem('engram_waitlist_joined', '1'); } catch (_) { /* private mode */ }
    setStatus("You're on the list! We'll be in touch at " + email + '.', true);
  } else {
    // Endpoint missing or unreachable: hand off to the user's mail client and
    // still confirm so the flow ends on a friendly note.
    mailtoFallback(email, note);
    setStatus('Could not reach the server, opening your email app instead. Or email us at ' + WAITLIST_EMAIL + '.', true);
  }

  if (emailInput) emailInput.value = '';
  if (noteInput) noteInput.value = '';
}

export function initWaitlist() {
  if (_wired) return;
  const dialog = $('waitlist-dialog');
  if (!dialog) return;
  _wired = true;

  $('btn-waitlist')?.addEventListener('click', () => openWaitlist());
  $('waitlist-close')?.addEventListener('click', () => dialog.close());
  // The send button is type="submit", so both button click and Enter-in-input
  // arrive here as a single submit event.
  dialog.querySelector('form')?.addEventListener('submit', (ev) => {
    ev.preventDefault();
    submit();
  });
}

export function openWaitlist() {
  initWaitlist();
  const dialog = $('waitlist-dialog');
  if (!dialog) return;
  setStatus('', false);
  if (!dialog.open) dialog.showModal();
  $('waitlist-email')?.focus();
}
