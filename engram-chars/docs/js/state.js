// state.js — pure state derivation from event stream.
//
// Replay engine is *deterministic*: given an event array and an index,
// rebuilding state means walking events 0..i and applying each one.
// No event handler does I/O or animation here — those are side effects of
// the renderer (render.js). Keeping state pure means scrubbing back/forward
// just reapplies events from scratch.

export function freshState(header) {
  const baseline = header?.baseline_ocean ?? { O:0.5, C:0.5, E:0.5, A:0.5, N:0.5 };
  return {
    npc_id:     header?.npc_id     ?? 'unknown',
    npc_name:   header?.npc_name   ?? 'Unknown',
    persona:    header?.persona    ?? '',
    config:     header?.config     ?? {},
    baseline_ocean:  { ...baseline },
    effective_ocean: { ...baseline },

    memories: [],          // {id, importance, source, text, tags}
    selected_ids: [],      // most recent retrieval_scored.selected_ids
    retrieval_threshold: header?.config?.retrieval_threshold ?? 15,

    session_window: [],    // [{player, npc}]
    summaries: [],         // strings
    facts: [],             // {kind:'asserted'|'revised'|'rejected', fact, old?}
    key_memories: [],      // {ids, count}

    turns: [],             // per-turn aggregates (see makeTurn)
    transcript: [],        // {who:'player'|'npc', text, mode?, turn}

    // ephemeral / animation hints (cleared by renderer once consumed)
    flash_ff: false,
    fresh_mem_id: null,
    fresh_fact_idx: null,
    fresh_key: false,

    // window-size cap from config
    window_size: header?.config?.session_window ?? 7,

    last_event_t: 0,
    current_turn: -1,
    session_ended: false,
  };
}

function makeTurn(turnNo, playerInput) {
  return {
    turn: turnNo,
    player_input: playerInput,
    threat: null,         // {is_threat, magnitude}
    retrieval: null,      // {mode, scored, threshold, selected_ids}
    mode: null,           // 'standard'|'fight_flight'|'instinct'
    contradiction: null,  // {stage, conflicts}
    response: null,       // {text, attempt}
    consolidated: null,   // {memory_id}
    duration_ms: null,
  };
}

// Apply one event. Returns the same state object (mutated for speed).
export function apply(state, ev) {
  state.last_event_t = ev.t ?? state.last_event_t;
  const p = ev.payload ?? {};

  switch (ev.type) {
    case 'session_init':
    case 'session_init_npc': {
      // Should already have been used to build state — but if it appears
      // mid-stream (multi-NPC), re-initialise from it.
      const next = freshState(p);
      Object.assign(state, next);
      break;
    }

    case 'turn_start': {
      state.current_turn = p.turn;
      state.turns.push(makeTurn(p.turn, p.player_input));
      state.transcript.push({ who: 'player', text: p.player_input, turn: p.turn });
      break;
    }

    case 'embedding_done':
      break;

    case 'threat_assessed': {
      const t = currentTurn(state);
      if (t) t.threat = { is_threat: !!p.is_threat, magnitude: p.magnitude ?? 0, reasoning: p.reasoning ?? '' };
      break;
    }

    case 'retrieval_scored': {
      const t = currentTurn(state);
      if (t) {
        t.retrieval = {
          mode: p.mode ?? 'scored',
          scored: p.scored ?? [],
          threshold: p.threshold ?? state.retrieval_threshold,
          selected_ids: p.selected_ids ?? [],
        };
      }
      state.selected_ids = p.selected_ids ?? [];
      break;
    }

    case 'mode_selected': {
      const t = currentTurn(state);
      if (t) t.mode = p.mode;
      break;
    }

    case 'fight_flight_applied': {
      state.flash_ff = true;
      break;
    }

    case 'contradiction_check': {
      const t = currentTurn(state);
      if (t) {
        // multiple stages may fire per turn (pre/post/re-rolled);
        // keep the latest, but track conflicts cumulatively
        const existing = t.contradiction ?? { stages: [], conflicts: [] };
        existing.stages.push(p.stage ?? 'check');
        if (p.conflicts && p.conflicts.length) existing.conflicts.push(...p.conflicts);
        existing.last_stage = p.stage ?? 'check';
        existing.last_text = p.text ?? '';
        t.contradiction = existing;
      }
      break;
    }

    case 'response_generated': {
      const t = currentTurn(state);
      if (t) t.response = { text: p.text ?? '', attempt: p.attempt ?? 1 };
      state.transcript.push({ who: 'npc', text: p.text ?? '', mode: t?.mode, turn: state.current_turn });
      break;
    }

    case 'consolidated': {
      const t = currentTurn(state);
      if (t) t.consolidated = { memory_id: p.memory_id };
      // also update the rolling session window
      if (t?.player_input != null && t?.response?.text != null) {
        state.session_window.push({ player: t.player_input, npc: t.response.text });
        if (state.session_window.length > state.window_size) {
          state.session_window.shift();
        }
      }
      break;
    }

    case 'memory_added': {
      state.memories.push({
        id: p.id, source: p.source ?? 'session',
        importance: p.importance ?? 0,
        text: p.text ?? '',
        tags: p.tags ?? null,
      });
      state.fresh_mem_id = p.id;
      break;
    }

    case 'summary_added': {
      state.summaries.push(p.summary ?? '');
      break;
    }

    case 'key_promoted': {
      state.key_memories = { ids: p.memory_ids ?? [], count: p.count ?? (p.memory_ids?.length ?? 0) };
      state.fresh_key = true;
      break;
    }

    case 'fact_asserted': {
      state.facts.push({ kind: 'asserted', fact: p.fact });
      state.fresh_fact_idx = state.facts.length - 1;
      break;
    }
    case 'fact_revised': {
      state.facts.push({ kind: 'revised', fact: p.fact, old: p.old });
      state.fresh_fact_idx = state.facts.length - 1;
      break;
    }
    case 'fact_rejected': {
      state.facts.push({ kind: 'rejected', fact: p.fact, old: p.old });
      state.fresh_fact_idx = state.facts.length - 1;
      break;
    }

    case 'profile_decay': {
      if (p.effective) state.effective_ocean = { ...p.effective };
      break;
    }

    case 'turn_end': {
      const t = currentTurn(state);
      if (t) t.duration_ms = p.duration_ms ?? null;
      // selected_ids glow until next turn — clear here so next turn starts fresh
      state.selected_ids = [];
      break;
    }

    case 'session_end':
    case 'session_end_npc':
      state.session_ended = true;
      break;
  }
  return state;
}

function currentTurn(state) {
  return state.turns[state.turns.length - 1] ?? null;
}

// Rebuild state from scratch through events[0..upto].
export function rebuild(events, upto, header) {
  const state = freshState(header);
  for (let i = 0; i <= upto && i < events.length; i++) {
    apply(state, events[i]);
  }
  return state;
}
