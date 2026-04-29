# Engram — Architecture

How the personality-parameterized memory pipeline works, mapped to the FDG '26 paper.

---

## 1. What this is

Each NPC is an `NPCAgent` (`src/engram/npc.py`) that owns:

- A static `NPCConfig` — name, persona, backstory lines, OCEAN profile, seed Prolog facts.
- A mutable `OCEANProfile` with five baseline trait values (0–1) plus five fight/flight deltas that decay each turn.
- A `MemoryManager` coordinating three stores:
  - **Session memory** — rolling 7-message window.
  - **Long-term memory** — OCEAN-biased two-line summaries of evicted batches.
  - **Key memory store** — top-25% memories promoted to a per-NPC SWI-Prolog fact base.
- Conversation history, turn counter, and a session buffer of memories formed this session.

Two entry points: `run_turn(player_input) -> str` and `end_session()`.

---

## 2. Per-turn pipeline (`run_turn`)

Six stages, mapped to paper §3.

### Stage 1 — Threat assessment (§3.1)

`pipeline/threat.py :: assess_threat`

1. Embed the player input once (this embedding is reused by stage 2).
2. Pull top-3 memories by **personality-weighted score** (`retrieve_top_scored`, no threshold) — same Eq. 1 formula as standard retrieval. The LLM sees the memories this NPC's tag/trait profile makes salient, not just the most semantically similar text.
3. Ask the LLM to return `{is_threat, threat_magnitude, reasoning}` as JSON, with the NPC's effective OCEAN values inlined.

The threat call **does not** decide the response mode anymore — it only returns the threat judgement. Mode is decided after retrieval scores are known (§3.3).

### Stage 2 + 3 — Scored retrieval drives mode selection (§3.2, §3.3)

`pipeline/retrieval.py` + `memory/manager.py :: retrieve`

The scoring formula (paper Eq. 1):

```
score = (RAG_similarity × 2) × Σ_t (t_mem_norm / t_agent) × importance
```

Implementation details (`manager.py :: _score`):

- `RAG_similarity` = cosine(query_embedding, memory.embedding)
- `t_mem_norm` = `tags.ocean[t] / 5` (normalises the 1–5 LLM tag to 0–1)
- `t_agent` = `profile.effective[t]`, floored at 0.05 to avoid divide-by-zero
- per-trait ratio capped at 5.0 to dampen runaway terms when `t_agent` is small
- `importance` = LLM-judged 1–10 integer

`MemoryManager.retrieve()` returns up to 5 memories with `score ≥ RETRIEVAL_THRESHOLD (15.0)` sorted descending. **No fallback** — empty list is meaningful.

Mode selection (in `npc.py :: run_turn`):

| Branch | `is_threat` | scored result | mode | retrieval used in prompt |
|---|---|---|---|---|
| Threat | true | (any) | `fight_flight` | scored (≥15) |
| Has relevant memory | false | non-empty | `standard` | scored (≥15) |
| No relevant memory | false | empty | `instinct` | `tag_retrieve` (6-D EventTags cosine, top-3) |

When `is_threat` is true, the agent also calls `profile.apply_fight_flight(magnitude)` which sets temporary deltas: `ΔN = +min(0.3, m·0.4)`, `ΔA = −min(0.2, m·0.25)`, `ΔE = −min(0.15, m·0.2)`. Deltas decay 10% per turn.

### Stage 4 — Response generation (§3.3)

See *Stage 5* below — the response is generated *first*, and then checked. This is the order the paper §3.4 actually specifies ("the system is sent the same prompt with the contradiction flagged" — i.e. a re-roll, not a pre-flag).

### Stage 5 — Post-response Prolog contradiction check + re-roll (§3.4)

`pipeline/consolidation.py :: check_contradictions`

After the LLM produces a response:

1. `extract_facts(response, ...)` — LLM extracts the factual claims the NPC just made.
2. Each candidate is normalised to Prolog form and queried via `KeyStore.check_contradiction`.
3. If conflicts surface AND the agent has **Openness < 0.5** → re-prompt once with the rejected response and the specific conflicts injected as `PRIOR ATTEMPT REJECTED`. The re-prompt asks the LLM to regenerate without re-asserting the contradiction.
4. If conflicts surface AND **Openness ≥ 0.5** → keep the response. Open NPCs are allowed to drift in dialogue; whether they actually update their beliefs is decided at `end_session`.

**Player-input claims are not checked during a turn.** They flow into the response naturally, get encoded in the session memory at stage 6, and are reconciled against the Prolog DB only at `end_session` via `post_session_fact_check` — which is the *only* place the Prolog DB is mutated.

### Stage 4 — Response generation (§3.3) [detail]

`pipeline/response.py :: generate_response`

Builds a sectioned prompt:

1. **TASK** — mode + threat status.
2. **CURRENT INPUT** — the raw player message.
3. **CONVERSATION HISTORY** — last 4 turns.
4. **CHARACTER BACKGROUND** — persona text.
5. **RELEVANT MEMORIES** — bullet list of retrieved memories.
6. **PERSONALITY** — `profile.describe()` plus per-trait behavioural instruction picked from `_TRAIT_HIGH/_LOW/_MID` based on the **effective** trait value.
7. **LONG-TERM SUMMARIES** — *standard mode only*. Last 3 entries from `LongTermMemory.summaries`.
8. **Mode addendum** — fight/flight or instinct rider.
9. **CONTRADICTION DETECTED** — listed conflicts; the LLM is told not to accept the new claim at face value.
10. **RULES** — stay in character, 2–4 sentences, no meta.

### Stage 6 — Memory consolidation (§3.5)

`pipeline/consolidation.py :: consolidate`

1. Combine into a single text: `"Player: <p> | <name>: <r>"`.
2. Embed and tag (`tag_event` returns an `EventTags`).
3. Build a `Memory(source="session")` and call `MemoryManager.add_memory()`.
4. Call `MemoryManager.add_turn(p, r)` which appends to `SessionMemory`. When the window overflows, the oldest 5 turns are evicted and `summarize_turns(evicted, profile)` produces an OCEAN-biased two-line summary that is appended to `LongTermMemory`.

### Bookkeeping

After stage 6: `profile.decay(0.1)` shrinks the fight/flight deltas, `turn_count++`, and the turn is appended to `history`.

---

## 3. End of session (`end_session`)

```
KeyStore.update(all_memories, profile)        ─► top 25% promoted, .pl rewritten
post_session_fact_check(session_memories)     ─► Openness-gated belief revision
save_state()                                   ─► state.json + memories.json
session_memories.clear()
```

`KeyStore.update`:
- Sorts all memories by `_personality_score` (the same OCEAN × importance formula minus the RAG term — there's no current query at promotion time).
- Takes `ceil(N × 0.25)` and writes them to `keystore.pl` as `key_memory(Id, Text).` plus any asserted facts.
- Reloads the file into the live SWI-Prolog runtime so subsequent `check_contradiction` queries see the new state.

`post_session_fact_check`:
- For each memory in this session, `extract_facts` → list of Prolog strings.
- For each fact: `keystore.check_contradiction`.
  - **Conflict + `profile.O ≥ 0.5`** → retract old, assert new (high-Openness accepts revision).
  - **Conflict + `profile.O < 0.5`** → skip (low-Openness keeps the existing belief).
  - **No conflict** → assert.

This is where the §3.4 mechanism actually mutates the fact base. The in-loop check during `run_turn` only flags conflicts to the LLM; it never rewrites memory.

---

## 4. Data pipeline diagram

### Per-turn (`run_turn`)

```
                   ┌──────────────────┐
player_input ────► │ llm.embed()      │ ──► query_embedding
                   └──────────────────┘             │
                                                    ▼
                   ┌──────────────────────────────────────────────┐
   §3.1            │ assess_threat()                              │
                   │  • retrieve_top_by_embedding (top-3, cosine) │
                   │  • LLM JSON judgement                        │
                   └──────────────────────────────────────────────┘
                                                    │
                            ThreatAssessment(is_threat, magnitude, reasoning)
                                                    ▼
                   ┌──────────────────────────────────────────────┐
   §3.2 + §3.3     │ Mode selection                               │
                   │                                              │
                   │ if is_threat:                                │
                   │   profile.apply_fight_flight() → ΔN/ΔA/ΔE    │
                   │   mode = "fight_flight"                      │
                   │   retrieved = scored_retrieve()              │
                   │                                              │
                   │ else:                                        │
                   │   retrieved = scored_retrieve()  (≥15 hard)  │
                   │   if retrieved: mode = "standard"            │
                   │   else:                                      │
                   │     mode = "instinct"                        │
                   │     retrieved = tag_retrieve()  (6-D tags)   │
                   └──────────────────────────────────────────────┘
                                                    │
                                                    ▼
                   ┌──────────────────────────────────────────────┐
   §3.4            │ check_contradictions()                       │
                   │  • extract_facts(player_input)               │
                   │  • for each → keystore.check_contradiction() │
                   │  → list[(new_fact, old_fact)]                │
                   └──────────────────────────────────────────────┘
                                                    │
                                                    ▼
                   ┌──────────────────────────────────────────────┐
   §3.3            │ generate_response()                          │
                   │   sections injected:                         │
                   │     • task / threat status / mode            │
                   │     • current input                          │
                   │     • last 4 history turns                   │
                   │     • persona + backstory                    │
                   │     • RELEVANT MEMORIES (retrieved)          │
                   │     • PERSONALITY (per-trait instructions)   │
                   │     • LONG-TERM SUMMARIES  (standard mode)   │
                   │     • mode addendum  (fight_flight/instinct) │
                   │     • CONTRADICTION DETECTED  (if any)       │
                   │     • rules                                  │
                   └──────────────────────────────────────────────┘
                                                    │
                                            response (str)
                                                    ▼
                   ┌──────────────────────────────────────────────┐
   §3.5            │ consolidate()                                │
                   │  • combined_text = "Player: … | NPC: …"      │
                   │  • llm.embed(combined_text)                  │
                   │  • tag_event(combined_text)                  │
                   │  • Memory(source="session") → MM.add_memory()│
                   │  • MM.add_turn(p, r) → may evict + summarise │
                   │      → LongTermMemory.add_summary()          │
                   └──────────────────────────────────────────────┘
                                                    │
                                                    ▼
                   profile.decay() ─► turn_count++ ─► history.append()
                                                    │
                                                    ▼
                                              return response
```

### End of session (`end_session`)

```
KeyStore.update(all_memories, profile)        ─► top 25% promoted
                                                   keystore.pl rewritten
                                                   live runtime reloaded
            │
            ▼
post_session_fact_check(session_memories)
   for each memory:
     extract_facts() → prolog strings
     for each fact:
       keystore.check_contradiction()
         ├── conflict + profile.O ≥ 0.5  →  retract old, assert new
         ├── conflict + profile.O <  0.5 →  reject  (low-O keeps belief)
         └── no conflict                 →  assert new
            │
            ▼
save_state()    ─► state.json   (deltas, history, turn_count)
                   memories.json (full corpus)
session_memories.clear()
```

### Where each artefact lives in memory and on disk

```
                       ┌────────────────────────────┐
 player input ─────►   │ NPCAgent                   │
                       │  ├─ history          (RAM) │ ◄── state.json
                       │  ├─ session_memories (RAM) │
                       │  └─ profile          (RAM) │ ◄── (deltas) state.json
                       │                            │
                       │  MemoryManager             │
                       │   ├─ all_memories ─────────┼──► memories.json
                       │   ├─ SessionMemory  (RAM)  │
                       │   ├─ LongTermMemory ───────┼──► longterm.json
                       │   └─ KeyStore   ───────────┼──► keystore.pl + SWI-Prolog
                       └────────────────────────────┘
```

---

## 5. Memory storage — what, when, where, in what form

### What gets stored

Every NPC produces three kinds of memory artefacts during its lifetime:

1. **Raw memories** — one `Memory` object per recorded experience.
   - **Backstory memories** — pre-loaded once, from `NPCConfig.backstory`, on first run only (`_init_backstory`).
   - **Session memories** — one per `run_turn`, encoding the player's input + NPC's reply as a single combined memory.
2. **Long-term summaries** — OCEAN-biased two-line strings summarising evicted dialogue batches.
3. **Prolog facts + key memories** — a SWI-Prolog `.pl` file containing the top-25% memories' text and the NPC's structured beliefs (`fact(...)`, `relationship(...)`, `believes(...)`).

Plus a small `state.json` for runtime housekeeping (history, turn count, fight/flight deltas).

### When each gets written

| Artefact | Trigger |
|---|---|
| Backstory memories | First run only — when no `state.json` exists yet for this NPC |
| Session memory | Every `run_turn`, in stage 6 (`consolidate`) |
| Long-term summary | When the 7-turn `SessionMemory` window overflows: oldest 5 turns get summarised → 1 summary appended |
| Prolog key memories | At `end_session` — `KeyStore.update()` re-ranks all memories and rewrites the `.pl` file with the top 25% |
| Prolog facts | (a) on first run, from `initial_facts`; (b) at `end_session`, from `post_session_fact_check` (Openness-gated) |
| `state.json` | At `end_session` |

### Where each lives on disk (`data/<npc_id>/`)

| File | Writer | Reader |
|---|---|---|
| `memories.json` | `MemoryManager.save_memories()` (called at end of session) | `MemoryManager.load_memories()` at agent init |
| `longterm.json` | `LongTermMemory.add_summary()` (on each eviction) | `LongTermMemory._load()` at init; **read at every standard-mode turn** and injected into the response prompt |
| `keystore.pl` | `KeyStore._write_pl()` (after every `update` or `assert_fact`) | `KeyStore.__init__()` consults; **read on every in-loop contradiction check and at end-of-session reconciliation** |
| `state.json` | `save_state()` | `_load_state()` at agent init |

The whole memory corpus is also held in **RAM** while the agent is alive (`MemoryManager.all_memories: list[Memory]`). On-disk JSON is the persistence layer; in-memory list is the working set. They are kept in sync at session boundaries (and after every `add_memory` call).

### What a `Memory` actually looks like

```json
{
  "id": "guard_backstory_0",
  "text": "You grew up in this town and have worked the docks since you were twelve.",
  "tags": {
    "emotion_valence": 0.1,
    "social_type": "cooperation",
    "threat_level": 0.0,
    "goal_relevance": 0.6,
    "novelty_level": 0.5,
    "self_relevance": 1.0,
    "importance": 7,
    "ocean": { "O": 1, "C": 5, "E": 2, "A": 3, "N": 2 }
  },
  "embedding": [0.0040, -0.0024, -0.0013, ... ],
  "source": "backstory",
  "timestamp": 1761234567.89,
  "score": 0.0
}
```

Field by field:

| Field | Type | Meaning |
|---|---|---|
| `id` | str | Unique. Backstory: `<npc_id>_backstory_<i>`. Session: UUID4. |
| `text` | str | Raw text. Backstory: a single sentence. Session: `"Player: <p> | <name>: <r>"`. |
| `tags` | `EventTags` | LLM-judged metadata. See below. |
| `embedding` | list[float] | Gemini embedding of `text`. ~768 floats per memory. |
| `source` | str | `"backstory"` or `"session"`. |
| `timestamp` | float | Unix epoch when the memory was created. |
| `score` | float | Last retrieval score (mutated each time the memory is surfaced — informational only). |

### What `EventTags` actually means

| Field | Range | Source |
|---|---|---|
| `emotion_valence` | [-1, 1] | LLM judgement |
| `social_type` | one of `solitude` / `conversation` / `cooperation` / `conflict` | LLM |
| `threat_level` | [0, 1] | LLM |
| `goal_relevance` | [0, 1] | LLM |
| `novelty_level` | [0, 1] | LLM |
| `self_relevance` | [0, 1] | LLM |
| `importance` | int 1–10 | LLM |
| `ocean` | `{O,C,E,A,N}` 1–5 ints | LLM — **not the NPC's personality**, but how strongly the *event* engages each trait dimension |

The crucial distinction: `tags.ocean` is **per-memory**, set once at encoding time, and never changed. It records "to fully process this event you'd lean on these trait dimensions." `OCEANProfile` is **per-agent** and represents the NPC itself. Retrieval scoring (Eq. 1) combines them via `tags.ocean[t] / profile.effective[t]`.

### What `keystore.pl` looks like

```prolog
% Engram KeyStore — auto-generated, do not edit by hand.

% key_memory(Id, Text).
key_memory('guard_backstory_5', 'You keep the storeroom key on you at all times — it is your responsibility.').
key_memory('a3f2e1d0-...', 'Player: Give me the storeroom key. Now. | Rico: I don''t even know you.').

% fact(NpcId, Subject, Predicate, Object).
fact(rico, sofia, status, alive).
relationship(rico, sofia, ally).
believes(docks_are_dangerous, true).
```

Key memories are denormalised — same text appears in `memories.json` too. The `.pl` form exists so SWI-Prolog can pattern-match across them (and so the file survives process restarts). Facts come from `extract_facts()` and the NPC's seeded `initial_facts`.

### What `longterm.json` looks like

```json
{
  "summaries": [
    "A stranger demanded the storeroom key, and Rico refused with hostility, recognising the threat to his charge. The encounter reinforced his vigilance — outsiders cannot be trusted near the docks.",
    "The harvest dinner invitation came warmly, but Rico kept his distance. Crowds remain a place where intentions hide."
  ]
}
```

One entry per eviction (every 5 evicted turns). Written in the NPC's voice, biased by personality.

### What `state.json` looks like

```json
{
  "turn_count": 12,
  "history": [
    {"player": "...", "npc": "..."},
    ...
  ],
  "profile_deltas": {"_dO": 0.0, "_dC": 0.0, "_dE": -0.05, "_dA": -0.08, "_dN": 0.12}
}
```

Conversation history is replayed into prompts (last 4 turns). Deltas restore the agent's fight/flight perturbation across process restarts so a multi-turn confrontation doesn't reset between runs.

---

## 6. Module map

```
src/engram/
├── config.py                 thresholds, model IDs (env-driven API key)
├── models.py                 dataclasses: OCEANProfile, EventTags, Memory,
│                              ThreatAssessment, NPCConfig
├── npc.py                    NPCAgent — orchestrator
│
├── llm/
│   ├── client.py             GeminiClient (generate / generate_json / embed)
│   └── tagging.py            tag_event, extract_facts, summarize_turns
│
├── memory/
│   ├── session.py            SessionMemory — rolling 7-turn window, evicts 5
│   ├── longterm.py           LongTermMemory — appended summaries, JSON-backed
│   ├── keystore.py           KeyStore — Prolog-backed key facts + key memories
│   └── manager.py            MemoryManager — wires the three stores; defines
│                              the personality-weighted retrieval scorer
│
└── pipeline/
    ├── threat.py             Stage 1 — assess_threat
    ├── retrieval.py          Stage 2 — scored_retrieve, tag_retrieve
    ├── consolidation.py      Stage 4 — check_contradictions
    │                         Stage 6a — consolidate (per-turn)
    │                         Stage 6b — post_session_fact_check
    └── response.py           Stage 5 — generate_response
```

Top-level entry points: `src/demo.py` (side-by-side two-agent CLI) and `eval/character_eval.ipynb` (CharacterEval-style scoring notebook).

---

## 7. Key constants (`src/engram/config.py`)

| Name | Value | Used by |
|---|---|---|
| `SESSION_WINDOW` | 7 | `SessionMemory` — Miller's 7±2 |
| `EVICT_BATCH` | 5 | `SessionMemory` — turns evicted per overflow → 1 summary |
| `RETRIEVAL_THRESHOLD` | 30.0 | Hard score gate in `MemoryManager.retrieve` (calibrated — see below) |
| `TOP_K_RETRIEVAL` | 5 | Top-K cap in `MemoryManager.retrieve` |
| `KEY_MEMORY_PERCENTILE` | 0.75 | Top 25% promoted in `KeyStore.update` (i.e. `1.0 - 0.75`) |
| `DECAY_RATE` | 0.1 | Per-turn fight/flight delta decay |
| `THREAT_MAX_TOKENS` | 200 | Caps stage-1 LLM call |

---

## 8. Personality data flow (where OCEAN actually matters)

The point of Engram is that personality affects **encoding and retrieval**, not just dialogue style. Concretely:

- **Tagging** (`tag_event`) — the LLM emits `ocean: {O, C, E, A, N}` 1–5 integers indicating which traits an *observer* needs to process this event. These are stored on the memory and never updated.
- **Threat assessment** (`assess_threat`) — the prompt includes the NPC's effective OCEAN so high-N NPCs are more likely to flag threats and high-A NPCs less likely.
- **Retrieval scoring** (`MemoryManager._score`) — `t_mem / t_agent` per trait. The division (rather than multiplication) is deliberate: it makes the agent's trait vector a *selector* across memories rather than a uniform amplifier. If we multiplied, every NPC would converge on the same maximally-tagged memories. Dividing means the per-trait contribution is normalised by agent strength, so each agent's profile produces a different ranking over the same corpus. Personality bias toward "more threat memories for high-N NPCs" is delivered by the *encoding* path (high-N agents flag more events as threats in stage 1 and at tagging time), not by re-amplifying them at retrieval. The 5.0 per-trait cap prevents very small `t_agent` from blowing up the score.
- **Tag retrieval (instinct)** — falls back to 6-D `EventTags` cosine when nothing crosses the threshold, so personality-relevant tags still drive selection.
- **Summarisation** (`summarize_turns`) — the prompt tells the LLM to summarise *as this NPC experienced it*, with per-trait tone instructions.
- **Per-trait response instructions** (`_trait_instruction` in `response.py`) — high/low/mid bands for each of O/C/E/A/N inject behavioural guidance into the prompt.
- **Fight/flight deltas** — temporary OCEAN perturbation on threat that decays back to baseline; affects every personality-using prompt for the next few turns.
- **Belief revision gate** (`post_session_fact_check`) — Openness ≥ 0.5 accepts revisions, < 0.5 rejects.

---

## 9. Threshold calibration (RETRIEVAL_THRESHOLD)

The threshold was raised from **15** to **30** based on the all-pairs score distribution in `eval/chareval_data/`.

Per-profile within-corpus score distribution (each memory used as a hypothetical query against the others):

| Profile | mean | p50 | p75 | p90 | p95 | max |
|---|---|---|---|---|---|---|
| Paranoid Guard | 113 | 122 | 134 | 158 | 166 | 181 |
| Friendly Merchant | 63 | 61 | 84 | 96 | 104 | 112 |
| Rigid Clerk | 76 | 74 | 98 | 121 | 129 | 138 |

These numbers are inflated relative to real player-vs-memory scores (within-corpus pairs share vocabulary, so cosine is high). Multiplying by ~0.5 gives a realistic estimate for player-input queries: Guard ~55, Merchant ~30, Clerk ~38.

At **threshold = 15**, every memory crossed the gate for every profile — the "hard threshold" was effectively a no-op. Instinct mode would essentially never fire once any memories existed.

At **threshold = 30**:
- Guard: well below average — almost every relevant memory still surfaces, instinct mode fires only on truly novel inputs.
- Merchant: near the median — selective gating, ~half of memories cross on a typical query, top-5 narrows further.
- Clerk: between mean and p25 — moderate gating.

This preserves the paper's intent: standard mode dominates when the agent has relevant memories; instinct mode meaningfully fires for novel/orthogonal inputs. The asymmetry across profiles is structural (low-trait agents inflate `t_mem / t_agent` ratios), not a bug — the threshold is a *minimum salience* floor, and each profile's salience baseline differs.

To re-calibrate against new data:

```python
PYTHONPATH=src python3 - <<'EOF'
# all-pairs score histogram per profile — see git history of this section
EOF
```

Tune so the threshold sits between p25 and p50 for the corpus's *least selective* profile (here: Merchant). That keeps standard mode reachable for everyone while still letting instinct mode trigger when relevance genuinely drops off.

---

## 10. Failure modes worth knowing

- **No SWI-Prolog installed.** `KeyStore` logs a warning and degrades — `check_contradiction` always returns `(False, "")`, `assert/retract` are no-ops. The rest of the pipeline still runs; you just lose contradiction handling.
- **LLM JSON parse failure.** `tag_event` falls back to default `EventTags`; `assess_threat` returns "no threat"; `generate_response` returns whatever string came back (possibly empty). The pipeline never raises.
- **Empty embedding.** `assess_threat` short-circuits with a non-threat fallback; `scored_retrieve` returns `[]` (which routes to instinct mode).
- **No memories at all** (e.g. fresh agent with no backstory). Standard mode never fires — you get instinct on every turn until backstory is loaded or memories accumulate.
