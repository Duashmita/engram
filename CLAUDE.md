# Engram

Personality-Parameterized Schema Memory for NPC Cognitive Diversity — research prototype submitted to FDG '26.

## What this is

A multi-stage memory pipeline for LLM-driven NPCs where personality (OCEAN Big Five) governs memory encoding, retrieval, and consolidation — not just dialogue style. The same event produces structurally different memory traces across NPCs with different personalities.

## Paper

`/Users/duashmi/Downloads/Engram.pdf` — read this before making any architectural decisions.

## Open architecture questions

See `QUESTIONS.md`. Do not assume answers to unresolved questions — ask the user.

## Pipeline (in order)

1. **Threat Assessment** — quick RAG cosine similarity pass; amygdala analogue
2. **Personality-Weighted Retrieval** — scoring formula: `(RAG_score × 2) × Σ(t_mem/t_agent) × importance`; top 5 memories with score ≥ 15
3. **Response Mode Selection** — Standard (score ≥ 15, no threat) / Fight-Flight (threat detected) / Instinct (score < 15, no threat)
4. **Prolog Contradiction Check** — SWI-Prolog via pyswip; LLM identifies contradictions, Prolog verifies formally
5. **LLM Dialogue Generation**
6. **Memory Consolidation** — tagging + tiered storage

## Memory tiers

- **Session**: rolling 7-message window
- **Long-term summaries**: OCEAN-biased 2-line summary generated every 5 evictions
- **Key memories**: top 25% by personality-weighted score → promoted to Prolog fact base

## OCEAN personality vector

Format: `ocean(O, C, E, A, N)` on a 0–1 continuous scale.

Evaluation profiles from paper:
- Paranoid Guard (high-N, low-A): `ocean(0.2, 0.5, 0.3, 0.2, 0.9)`
- Friendly Merchant (high-E, high-A): `ocean(0.5, 0.5, 0.9, 0.8, 0.2)`
- Rigid Clerk (low-O, high-C): `ocean(0.1, 0.9, 0.3, 0.5, 0.4)`

## Key behaviors to preserve

- Personality affects memory **encoding**, not just dialogue output
- High-N NPCs flag disproportionately more threats
- High-E NPCs prefer socially-tagged memory retrieval
- Low-O NPCs resist belief revision after contradictions
- Fight/Flight mode produces temporary OCEAN deltas that decay back to baseline
- Openness score governs whether post-session contradictions are accepted or rejected

## Stack (to be confirmed via QUESTIONS.md)

- Language: Python
- Logic engine: SWI-Prolog via `pyswip`
- LLM/embedding provider: TBD
- Vector store: TBD
- Persistence: TBD

## Do not

- Apply personality only at the output/dialogue layer — it must affect encoding and retrieval
- Hardcode personality behavior in prompts — it must flow from the OCEAN vector
- Skip Prolog validation when generating facts from key memories
