/**
 * animations.js, Maps engram pipeline events → character animations.
 *
 * Call handleEvent(state, prevState, char) after every state.js apply() call.
 * Returns the animation name that was triggered (or null).
 */

// Personality → idle animation name (matches clip names downloaded from Meshy)
function personalityIdleAnim(ocean) {
  if (!ocean) return 'idle';
  if (ocean.N >= 0.7) return 'crouching';
  if (ocean.E >= 0.7) return 'wave';
  if (ocean.A <= 0.3) return 'combat';
  if (ocean.O >= 0.7) return 'looking';
  return 'idle';
}

// How long (ms) to hold a one-shot animation before returning to idle.
const ONESHOT_DURATION = {
  getting_hit:  900,
  scared:       1200,
  surprised:    1000,
  punch:        700,
  shaking:      1000,
  nodding:      800,
  wave:         1500,
  clapping:     2000,
  crying:       3000,
  laughing:     2000,
  dying:        4000,
};

// Keep track of the current mode so we don't spam animation changes.
let _currentMode = null;
let _fightFlightActive = false;

export function handleEvent(state, prevState, char) {
  if (!char) return null;

  // ── character_action tool call (highest priority) ────────────────────────
  if (state.character_action) {
    const { action, intensity } = state.character_action;
    const once = action in ONESHOT_DURATION;
    char.playAnim(action, { once, fade: 0.15 });
    // Fight/flight particles on hit
    if (action === 'hit' || action === 'getting_hit') {
      char.setParticles(true);
      setTimeout(() => char.setParticles(false), 1200);
    }
    state.character_action = null;
    return action;
  }

  // ── fight/flight applied → combat aura ───────────────────────────────────
  if (state.flash_ff && !_fightFlightActive) {
    _fightFlightActive = true;
    char.playAnim('combat', { loop: true, fade: 0.2 });
    char.setParticles(true);
    return 'combat';
  }

  // ── Pipeline mode changes ─────────────────────────────────────────────────
  const mode = state.turns?.at(-1)?.mode;
  if (mode && mode !== _currentMode) {
    _currentMode = mode;

    switch (mode) {
      case 'fight_flight':
        char.playAnim('combat', { loop: true, fade: 0.2 });
        char.setParticles(true);
        return 'combat';

      case 'instinct':
        char.setParticles(false);
        _fightFlightActive = false;
        char.playAnim('looking', { loop: true, fade: 0.3 });
        return 'looking';

      case 'standard':
        char.setParticles(false);
        _fightFlightActive = false;
        _setPersonalityIdle(state, char);
        return 'idle';
    }
  }

  // ── Per-turn stage events ─────────────────────────────────────────────────
  const lastTurn = state.turns?.at(-1);

  // Threat detected mid-turn
  if (lastTurn?.threat?.is_threat && !state.flash_ff) {
    char.playAnim('scared', { once: true, fade: 0.2 });
    return 'scared';
  }

  // Response generated → talking
  if (state.fresh_mem_id && lastTurn?.response) {
    // A new memory was just consolidated → talking + nodding
    char.playAnim('talking', { loop: true, fade: 0.2 });
    // Brief nodding acknowledgment after talking settles
    setTimeout(() => {
      if (_currentMode === 'standard') char.playAnim('nodding', { once: true, fade: 0.3 });
    }, 2000);
    return 'talking';
  }

  // Contradiction detected → head shake
  if (lastTurn?.contradiction?.length > 0) {
    char.playAnim('shaking', { once: true, fade: 0.15 });
    return 'shaking';
  }

  // Turn start → thinking while waiting
  if (lastTurn && !lastTurn.response && !lastTurn.mode) {
    char.playAnim('thinking', { loop: true, fade: 0.3 });
    return 'thinking';
  }

  // ── Turn end, return to personality idle ─────────────────────────────────
  if (!state.turns?.at(-1)?.turn || state.session_ended) {
    _setPersonalityIdle(state, char);
    return 'idle';
  }

  return null;
}

function _setPersonalityIdle(state, char) {
  const ocean = state.effective_ocean || state.baseline_ocean;
  if (!ocean) { char.playAnim('idle', { loop: true }); return; }
  const anim = personalityIdleAnim(ocean);
  char.playAnim(anim, { loop: true, fade: 0.5 });
}

/** Call when the session first starts to set the baseline idle. */
export function setInitialIdle(state, char) {
  _currentMode = null;
  _fightFlightActive = false;
  _setPersonalityIdle(state, char);
}
