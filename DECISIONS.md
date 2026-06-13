# Engram — Memory Architecture Decisions

The non-generic ideas — formulas, gates, and mechanisms specific to personality-parameterised memory. Skipping the boilerplate (language, storage backend, embedding model).

---

## Retrieval scoring (the central formula)

- **Eq. 1:**
  ```
  score = (RAG_sim × 2) × Σ_t (t_mem_norm / t_agent) × importance
  ```
  RAG cosine, summed per-trait OCEAN ratio, LLM importance — multiplied, not added. Each term must be non-trivial for a memory to surface.

- **Divide, don't multiply.** `t_mem / t_agent` makes the agent's personality a *selector* over memories. Multiplying would converge every NPC on the same maximally-tagged memories; dividing means each profile produces a different ranking over the same corpus.

- **`t_agent` floored at 0.05.** A perfectly calm NPC (N=0) would divide by zero against any neuroticism-tagged memory. The floor preserves the formula's intent without crashing.

- **Per-trait ratio capped at 5.0.** Prevents very small `t_agent` from blowing the score up; keeps the OCEAN term bounded relative to RAG and importance.

- **`t_mem_norm = tags.ocean[t] / 5`.** Memory tags live on a 1–5 integer scale; agent OCEAN is 0–1 continuous. The `/5` brings them to the same scale before the ratio.

- **Hard threshold, no fallback.** `RETRIEVAL_THRESHOLD = 30.0`. Top 5 by score, descending, ≥ threshold. Empty result is *meaningful* — it routes to instinct mode rather than degrading to "least bad" memories.

- **Threshold calibrated per-corpus.** Set between p25 and p50 of the least selective profile's score distribution. 30 came from the all-pairs distribution across Guard / Merchant / Clerk; the earlier value of 15 made every memory cross for every profile, killing instinct mode.

---

## Two OCEAN vectors, not one

The most important conceptual split in the system:

- **Agent OCEAN** (`OCEANProfile`, 0–1 continuous, mutable): who the NPC *is*. Carries decaying fight/flight deltas.
- **Memory OCEAN** (`tags.ocean`, 1–5 int, immutable): which trait dimensions an *observer* needs to process this event. Set once at encoding time, never updated.

If these were collapsed, retrieval scoring would reduce to "remember things that match your personality," and the encoding/retrieval distinction the paper rests on disappears.

---

## Mode selection happens *after* retrieval

| `is_threat` | scored retrieval | mode | memories used in prompt |
|---|---|---|---|
| true | (any) | `fight_flight` | scored (≥30) |
| false | non-empty | `standard` | scored (≥30) |
| false | empty | `instinct` | 6-D tag cosine, top-3 |

Threat detection decides *threat*, not response mode. Mode is a function of *(threat, do we have anything relevant to say?)* — the same retrieval pass drives both standard and fight/flight prompts.

**Instinct mode falls back to 6-D EventTags cosine**, not raw RAG. When no memory carries enough personality-weighted salience, retrieval drops the OCEAN term entirely and routes by tag similarity. This is deliberate: instinct = pre-personalised reflexes.

---

## Threat assessment is personality-informed by construction

- Stage 1 retrieves the top-3 memories by the **same Eq. 1 scoring**, not by raw cosine.
- The NPC's effective OCEAN is inlined in the threat prompt.

Effect: high-N NPCs surface more threat-tagged memories in their context window, biasing the LLM toward `is_threat=true` *without* a separate "be paranoid" instruction. The bias lives in retrieval, not in a prompt knob.

---

## Fight / flight as temporary OCEAN perturbation

- LLM judges threat magnitude; deltas are applied to the agent's OCEAN with hard caps:
  - `ΔN = +min(0.3, m·0.4)`
  - `ΔA = −min(0.2, m·0.25)`
  - `ΔE = −min(0.15, m·0.2)`
- **Multiplicative decay at 0.1/turn**, ≈20-turn half-life.
- Deltas persist across process restarts (`state.json`).

Crucially: deltas feed back into Eq. 1 via `t_agent`, so a frightened NPC retrieves a *different* set of memories on the next turn than a calm version of the same NPC. The perturbation isn't just dialogue tone — it changes what the agent remembers.

---

## Memory tier promotion gates

Three tiers, three different write triggers:

| Tier | Write trigger | Source data | Personality role |
|---|---|---|---|
| Session (7-turn window) | Every `run_turn` | Raw `"Player: … | NPC: …"` | None at write time |
| Long-term summaries | Every 5 evicted turns → 1 summary | Evicted session turns | LLM summarises *as this NPC experienced it* |
| Key memories + Prolog facts | `end_session` only | All-time memory pool | Re-ranked by Eq. 1 (RAG term dropped — no current query), top 25% promoted |

- **Batched eviction (5 → 1 summary), not FIFO.** A summary covers a mini-arc; per-turn summaries would be incoherent and burn LLM calls.
- **Long-term summaries grow unbounded.** Never re-summarised.
- **Key pool = every memory ever stored**, not just current session. A late-game event can dislodge a backstory memory if it scores higher.
- **Promotion uses post-decay OCEAN.** Promotion reflects the NPC's calmer, considered state — not a momentarily frightened version.

---

## Prolog as the belief layer

- **DB only mutated at `end_session`**, via `post_session_fact_check`. In-loop contradiction checks during a turn *flag* conflicts to the LLM but never write. This separates conversation from belief revision.
- **Player-input claims aren't checked mid-turn.** They flow into the response naturally, get encoded as session memory, and only get reconciled against Prolog at session end.
- **Openness gates revision:**
  - `O ≥ 0.5` → retract old, assert new.
  - `O < 0.5` → reject new, keep old.

  This is where the paper's "low-O NPCs resist belief revision" behaviour is mechanised. It's a belief-update gate, not a dialogue style choice.

- **Re-roll on contradiction also gated by Openness.** If the NPC's response contradicts existing facts AND `O < 0.5`, the response is regenerated once with the conflict flagged. High-O NPCs are allowed to drift in dialogue; whether they *commit* the drift to memory is still decided at session end.

- **Three fixed predicates: `fact`, `relationship`, `belief`.** LLM outputs JSON → Python converts to Prolog atoms. Raw-Prolog-from-LLM was rejected as fragile.

---

## Where personality enters the memory pipeline

If any of these collapse to "personality only at the dialogue layer," the architecture is gone:

1. **Encoding** — `tag_event` emits per-memory `tags.ocean`.
2. **Threat retrieval** — top-3 by Eq. 1, not raw cosine.
3. **Threat judgement** — effective OCEAN inlined in prompt.
4. **Standard retrieval** — Eq. 1 with the divide.
5. **Instinct fallback** — 6-D tag cosine when no memory crosses threshold.
6. **Summarisation** — LLM told to summarise as the NPC experienced it, per-trait tone.
7. **Fight/flight deltas** — temporary OCEAN perturbation feeds back into retrieval.
8. **Key promotion** — Eq. 1 (minus RAG) used to rank the all-time pool.
9. **Belief revision** — Openness gates retract/assert at session end.
10. **Re-roll** — Openness gates whether a contradicting response is regenerated.

Eight of these ten happen *before* the response prompt is built.
