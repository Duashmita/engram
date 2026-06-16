// waitlist.js
// Two-step "Join waitlist" dialog:
//   Step 1 — email only. Submitting registers the email and switches the dialog
//            to the thank-you panel (no form left on screen).
//   Step 2 — a thank-you message plus an OPTIONAL free-text note. The note is
//            sent on its own submit and upserted onto the already-registered
//            email row (backend add_waitlist).
//
// Wiring: app.js `import { initWaitlist }` and call it once on startup (it is
// idempotent). openWaitlist() opens the dialog and always resets to step 1.
//
// Defensive by design: the `${BACKEND_URL}/waitlist` endpoint may be down. On
// any failure we fall back to a mailto link so no signup is ever lost, and the
// flow still ends on the thank-you step.

import { BACKEND_URL } from '../config.js';

const WAITLIST_EMAIL = 'asdua@ucsc.edu';

let _wired = false;
let _email = '';   // remembered between step 1 and step 2

function $(id) {
  return document.getElementById(id);
}

function setStatus(id, msg, ok) {
  const el = $(id);
  if (!el) return;
  el.textContent = msg || '';
  el.classList.toggle('ok', !!ok);
}

function showStep(step) {
  const form = $('waitlist-form');
  const thanks = $('waitlist-thanks');
  if (form)   form.style.display   = step === 1 ? '' : 'none';
  if (thanks) thanks.style.display = step === 2 ? '' : 'none';
}

async function postWaitlist(email, note) {
  try {
    const res = await fetch(`${BACKEND_URL}/waitlist`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, note: note || '' }),
    });
    return res.ok;
  } catch (_) {
    return false;
  }
}

function mailtoFallback(email, note) {
  const body = 'Please add me to the Engram waitlist.\n\nEmail: ' + email
    + (note ? '\nWhat I am building: ' + note : '');
  const href = 'mailto:' + WAITLIST_EMAIL
    + '?subject=' + encodeURIComponent('Engram waitlist')
    + '&body=' + encodeURIComponent(body);
  try {
    window.location.href = href;
  } catch (_) { /* popup blockers etc. */ }
}

// Step 1: register the email, then advance to the thank-you step. We advance
// even on a network failure (the mailto fallback captures the signup), so the
// flow always ends on a friendly note.
async function submitEmail() {
  const emailInput = $('waitlist-email');
  const sendBtn = $('waitlist-send');
  const email = (emailInput?.value || '').trim();

  if (!email || !email.includes('@')) {
    setStatus('waitlist-status', 'Please enter a valid email address.', false);
    emailInput?.focus();
    return;
  }

  if (sendBtn) { sendBtn.disabled = true; sendBtn.textContent = 'Joining…'; }
  setStatus('waitlist-status', '', false);

  const joined = await postWaitlist(email, '');

  if (sendBtn) { sendBtn.disabled = false; sendBtn.textContent = 'Join our waitlist'; }

  if (joined) {
    // Remember the signup so the in-chat auto-prompt never nags this user again.
    try { localStorage.setItem('engram_waitlist_joined', '1'); } catch (_) { /* private mode */ }
  } else {
    // Endpoint unreachable: hand off to the user's mail client so the signup
    // isn't lost, then still advance to the thank-you step.
    mailtoFallback(email, '');
  }

  _email = email;
  if (emailInput) emailInput.value = '';
  showStep(2);
  $('waitlist-note')?.focus();
}

// Step 2: optional note, upserted onto the email row. Best-effort; an empty box
// just closes the dialog.
async function submitNote() {
  const noteInput = $('waitlist-note');
  const note = (noteInput?.value || '').trim();
  const sendBtn = $('waitlist-note-send');

  if (!note) {
    $('waitlist-dialog')?.close();
    return;
  }

  if (sendBtn) { sendBtn.disabled = true; sendBtn.textContent = 'Sending…'; }
  const ok = await postWaitlist(_email, note);
  if (sendBtn) { sendBtn.disabled = false; sendBtn.textContent = 'Send'; }

  if (ok) {
    setStatus('waitlist-thanks-status', 'Got it, thanks for the details!', true);
  } else {
    mailtoFallback(_email, note);
    setStatus('waitlist-thanks-status', 'Sent via email instead. Thanks!', true);
  }
  if (noteInput) noteInput.value = '';
}

export function initWaitlist() {
  if (_wired) return;
  const dialog = $('waitlist-dialog');
  if (!dialog) return;
  _wired = true;

  $('btn-waitlist')?.addEventListener('click', () => openWaitlist());
  $('waitlist-close')?.addEventListener('click', () => dialog.close());
  $('waitlist-thanks-close')?.addEventListener('click', () => dialog.close());

  // Each step is its own <form>; the submit buttons are type="submit", so both
  // a click and Enter-in-input arrive as a single submit event.
  $('waitlist-form')?.addEventListener('submit', (ev) => { ev.preventDefault(); submitEmail(); });
  $('waitlist-thanks')?.addEventListener('submit', (ev) => { ev.preventDefault(); submitNote(); });
}

export function openWaitlist() {
  initWaitlist();
  const dialog = $('waitlist-dialog');
  if (!dialog) return;
  setStatus('waitlist-status', '', false);
  setStatus('waitlist-thanks-status', '', false);
  showStep(1);
  if (!dialog.open) dialog.showModal();
  $('waitlist-email')?.focus();
}
