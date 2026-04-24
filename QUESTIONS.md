# Engram Architecture Questions

Questions to resolve before implementation. Answer inline or mark `[DECIDED]` with your choice.

---

## LLM & Embeddings

1. **Which LLM provider/model?** OpenAI, Anthropic, local (Ollama)? The paper doesn't specify. 
gemini api calls

2. **Which embedding model?** For cosine similarity RAG — OpenAI `text-embedding-3-small`, `sentence-transformers`, something else?
Use your best judgement

3. **Is threat assessment a separate LLM call** from dialogue generation, or one combined prompt that returns both threat flag + response?
seperate call with limited tokens depending on what we need

4. **How is "importance" scored?** What does the LLM-as-judge prompt look like — does it return a 1–10 number, a 1–100 number? What's the exact scale?
1-10

---

## OCEAN Tagging on Memories

5. **What scale are the per-memory OCEAN scores on?** The example in Figure 1 shows `"O": 4, "C": 4, "E": 3, "A": 3, "N": 2` — is that 1–5? 1–10? 0–1?
1-5

6. **What scale is the NPC's personality vector on?** Table 1 evaluation profiles use `ocean(0.2, 0.5, 0.3, 0.2, 0.9)` — so 0–1. Are the memory OCEAN scores normalized to the same scale before applying the formula?
yes

7. **What exactly is the LLM tagging?** The paper mentions: threat level, social type, emotional valence, goal relevance, novelty — are these numeric or categorical? Separately there are personality tags like `["responsible", "dutiful"]` — are these two different tagging passes?
tags: Openness to Experience – curiosity, creativity, willingness to try new things
Conscientiousness – organization, discipline, dependability, attention to detail
Extraversion – sociability, assertiveness, energy in social situations
Agreeableness – cooperativeness, empathy, harmony-seeking
Neuroticism – emotional stability, anxiety, stress reactivity

8. **The scoring formula** `score = (RAG × 2) × Σ(t_mem/t_agent) × importance` — is `t_mem` the per-trait OCEAN score stored on the memory, and `t_agent` the NPC's current trait value for that same trait? Is the sum over all 5 OCEAN traits?
t_agent is the inputed value of OCEAn by the user, T_mem is the score for that specific memory, every memory has a tag from each OCEAN Feature, and optionally the tags.
---

## Threat Assessment

9. **What does threat assessment do with cosine similarity?** Does it embed the player message, compare against all stored memories, and feed the top results to an LLM that decides threat? Or is it a direct LLM call on the raw message? direct llm call

10. **Is the threat flag binary (TRUE/FALSE)** or a score with a threshold? The Figure 1 example shows `"threat": FALSE`. binary

11. **What counts as "context to past experiences"** for the fight/flight decision — is it the top retrieved memories, or a specific set? top retreived, but the prompt mentions that it's flight or flight

---

## Response Modes

12. **Fight/Flight OCEAN deltas** — are these LLM-generated per encounter (e.g., "this encounter should shift N by +2") or hardcoded deltas per event type? llm as judge, but a max of 20%

13. **Decay mechanism** — how does personality return to baseline? Fixed decrement per interaction, a half-life formula, or something else? Over how many turns? 2 normsl messages in a row

14. **Instinct mode tag-based matching** — does the incoming player message get tagged first, then matched against key memory tags? Or is it keyword/embedding based? keyword embedding

---

## Prolog Contradiction Checking

15. **What schema are Prolog facts stored in?** E.g., `relationship(rico, sofia, ally)`, `knows(npc, fact)`? Fixed predicate types or freeform? relationships, game facts, story facts

16. **How does the LLM generate Prolog facts?** Is there a prompt that instructs it to output valid Prolog syntax, which then gets validated by pyswip? yes propmts 

17. **Is there a pre-seeded Prolog knowledge base** (backstory, world facts, relationships) loaded at NPC creation, or is it built entirely from in-game conversation? pre seeded

18. **When a contradiction is resolved post-session**, what happens to the old Prolog fact — retracted, flagged, or overwritten? depends on how open the character is in OCEAN Scoring

---

## Memory Storage & Persistence

19. **Where are memories stored?** In-memory Python dict, SQLite, a vector DB (ChromaDB, Pinecone, Qdrant)? json file

20. **Is there cross-session persistence?** Do NPCs remember things from previous game sessions, or does each session start fresh? they do yes, they remember

21. **The 7-message session window** — is this 7 player messages only, 7 NPC messages only, or 7 total turns (player + NPC counted as 1)? 7 total

22. **"Every five messages evicted"** — when session memory overflows, do exactly 5 get summarized and evicted at once? Or FIFO where the oldest is dropped and summarized individually? yur call

---

## Key Memory Promotion

23. **Top 25% of what pool?** Every memory ever stored, or only memories from the current session?
every memory

24. **When does promotion happen?** End of every interaction, session end, or whenever a new memory is saved? end of every section

---

## Multi-NPC & World State

25. **Are multiple NPCs running simultaneously?** Is there a shared event log or world state all NPCs can read? no multi NPCs


26. **Can NPCs share or communicate memories?** E.g., if NPC A tells NPC B something, does NPC B form a memory of being told that?
no npcs communicate.
---

## Interface & Integration

27. **What's the user-facing interface?** CLI, REST API, Python library, or direct game engine integration (Unity, Godot, Unreal)?
git pages, with a visual representation of memory active

28. **Is there a frontend/UI** at all, or is this purely backend? yes to frontend and ui

29. **How is an NPC "created"?** Config file, function call, character sheet format? the number entered by user when they start one

---

## Summarization

30. **"Two-line summary"** — literally two sentences, or approximate? Does the LLM get explicit instruction to produce exactly 2 lines?
yes

31. **Are long-term summaries ever re-summarized** as they accumulate, or do they grow unbounded?
no

---

## Architectural Decisions

### Module structure
**Decision:** Modular package at `src/engram/` with submodules: `llm/` (Gemini client + tagging), `memory/` (session, longterm, keystore, manager), `pipeline/` (threat, retrieval, response, consolidation), plus `models.py`, `config.py`, `prolog_engine.py`, `npc.py`. Demo at `src/demo.py`.
**Why:** Testability. Each pipeline stage is a standalone function — threat assessment, retrieval, and consolidation can be unit-tested without running the full NPC loop.

### Multi-agent architecture
**Decision:** Each `NPCAgent` is fully independent (own memory, Prolog KB, OCEAN vector). Pipeline stages are separate targeted LLM calls — threat (≤200 tokens), tagging, dialogue generation. No shared state between NPCs.
**Why:** This is the multi-agent cognitive architecture the paper describes. NPCs are cognitively isolated so their divergence comes purely from personality, not shared state contamination.

### Embedding backend
**Decision:** Gemini `text-embedding-004` for full text embeddings on every Memory object. 6D tag vectors used only for instinct-mode retrieval and OCEAN storage-probability. Real embeddings drive primary RAG.
**Why:** Paper implies semantic retrieval ("cosine similarity pass"). 6D tag vectors are coarse approximations. Real embeddings capture meaning the tags miss.

### Scoring formula edge case — division by zero
**Decision:** Floor `t_agent` at 0.05 before computing `t_mem/t_agent`. Cap each ratio at 5.
**Why:** A perfectly calm NPC (N=0.0) would produce division-by-zero for any memory with a neuroticism tag. Floor of 0.05 preserves the formula's intent without crashing.

### Threat assessment: two-step
**Decision:** Embed player input → cosine-sim top-3 memories → LLM call with that context → `ThreatAssessment`.
**Why:** Makes threat detection personality-informed without hardcoding it. High-N NPCs surface more threat-tagged memories in their top-3, biasing the LLM toward threat without a separate personality-awareness prompt.

### Fight/flight OCEAN deltas
**Decision:** `_dN = min(0.3, magnitude × 0.4)`, `_dA = -min(0.2, magnitude × 0.25)`, `_dE = -min(0.15, magnitude × 0.2)`. Multiplicative decay at rate 0.1 per turn (≈20-turn half-life).
**Why:** Psychologically grounded, no extra LLM call, directly tied to threat magnitude so more severe threats produce larger personality shifts.

### Prolog schema
**Decision:** Three fixed predicates: `fact(NpcId, Subject, Predicate, Object)`, `relationship(NpcId, Entity1, Relation, Entity2)`, `belief(NpcId, Claim, TruthValue)`. LLM outputs JSON → Python converts to Prolog atoms → pyswip asserts.
**Why:** Fixed schema keeps contradiction checking tractable. LLM outputting raw Prolog syntax is fragile; JSON→Prolog conversion in Python is deterministic and validated.

### Contradiction resolution and Openness
**Decision:** If `O ≥ 0.5`, retract old conflicting fact and assert new. If `O < 0.5`, reject new fact, keep old.
**Why:** Directly implements the paper's "low-O NPCs resist belief revision" behavior. Openness score governs belief update in a mechanistic, auditable way.

### Storage backend
**Decision:** JSON files per NPC in `data/{npc_id}/`. `session.json`, `longterm.json`. Prolog facts in `keystore.pl` loaded via `prolog.consult()`.
**Why:** Zero extra infrastructure. JSON is human-readable for debugging. Prolog file format is natural for pyswip.

### Session window and eviction
**Decision:** 7 total turns (player+NPC = 1 turn). When window exceeds 7, oldest 5 evicted as a batch → one OCEAN-biased 2-line summary LLM call.
**Why:** Batch eviction keeps summaries semantically coherent (5 turns ≈ a mini-arc of conversation) and minimizes LLM overhead vs. per-message eviction.

### Key memory promotion
**Decision:** Session end only. Pool = all memories ever stored. Top 25% by personality-weighted score. Rescored with post-decay OCEAN values before promotion.
**Why:** Mid-turn promotion would add latency and Prolog reload overhead. Post-decay OCEAN values mean the promotion reflects the NPC's calmer, considered state.

### Tagging: single LLM call
**Decision:** One LLM call returns a single JSON with both semantic tags (emotion_valence, social_type, threat_level, goal_relevance, novelty_level, self_relevance) and OCEAN affinity (O/C/E/A/N on 1–5).
**Why:** Reduces LLM calls from 2 to 1 per memory stored. The LLM has full context to rate both dimensions simultaneously.

### NPC creation
**Decision:** `NPCConfig` dataclass in Python. `NPCAgent(config, llm_client, data_dir)`. No config file format.
**Why:** Type-checked, IDE-autocompleted, no YAML/JSON parsing layer to maintain. Research prototype — keeping it simple is correct.

### Demo format
**Decision:** Terminal output with clear section headers, side-by-side text comparison of two NPCs. No Jupyter widgets or HTML panels.
**Why:** Reproducible in any environment (no Colab, no browser). Paper reviewers can run `python src/demo.py` and see results immediately.