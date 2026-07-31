# Changes

Log of code changes made in this repo, with the reasoning behind each —
newest first. Each entry covers: what changed, why it was needed, and how
it works.

---

## Added Kimi K3 as a hosted bench candidate (code only, no key yet)

**Files:** `bench/clients.py`, `bench/latency_bench.py`

**Why it was needed:** Kimi K3's open weights shipped but are still not
locally runnable on this hardware (see the entry below) — same shape of
problem as DiffusionGemma, same fix as was used for Mercury: use the
first-party hosted API instead of self-hosting.

**What changed:**
- `bench/clients.py`: added `KimiClient`, hitting Moonshot AI's
  OpenAI-compatible REST API (`POST https://api.moonshot.ai/v1/chat/completions`,
  `Authorization: Bearer $MOONSHOT_API_KEY`) via stdlib `urllib` — same
  pattern as `MercuryClient`, no new SDK dependency.
- `bench/latency_bench.py`: registered `kimi-k3` as a candidate, updated
  the module docstring to mention `MOONSHOT_API_KEY`.

**Note on cost:** Kimi K3 API pricing runs $3/M input, $15/M output tokens
— notably higher per-token than the other candidates here. Immaterial for
this benchmark's tiny prompts (a handful of short calls), but worth
knowing before running it at any larger scale.

**How it's verified working:** Syntax-checked (`ast.parse`). First real
run attempt (`--candidates kimi-k3 --runs 2`) failed on every prompt with
`HTTP 429`. Checked the raw response body directly (bypassing the client's
generic exception handling) to see the actual reason: `"Your account ...
is suspended due to insufficient balance, please recharge your account or
check your plan and billing details"` — auth succeeded (`Msh-Gid: free`
header present), this is a billing gate, not a bad key or rate limit.
Unlike Mercury/Inception, Moonshot doesn't grant free trial tokens on
signup — needs an account top-up before any call works. User added funds;
see the two entries below for the real run.

---

## Kimi K3 — thinking can't be disabled, only tuned down (`reasoning_effort`)

**Why it was needed:** After the account top-up, the first real run
(`--candidates kimi-k3 --runs 2`, default settings) came back at
**26,249ms/turn** — memory_tagging alone hit 36.5s median. That's the same
symptom class as the Gemini-thinking-truncation bug (below): unbudgeted
internal reasoning eating the latency. Checked whether it could be turned
off the same way (`thinking: {"type": "disabled"}` for Claude,
`thinking_budget=0` for Gemini).

**What was found (web search):** kimi-k3 cannot disable thinking at all —
it's a permanent architectural feature, not a default that happens to be
on. The only control is a top-level `reasoning_effort` field
(`low`/`high`/`max`, default `max`). Verified directly: a raw call with
`reasoning_effort: "low"` still returned a populated `reasoning_content`
field (81 reasoning tokens) alongside the answer — confirming thinking
runs even at the lowest tier, just less of it.

**What changed:** `bench/clients.py` — `KimiClient.generate()` now always
sends `"reasoning_effort": "low"`, matching the intent of the
`_DISABLE_THINKING_MODELS` handling in `ClaudeClient` and the
`thinking_budget=0` config in `GeminiClient`: none of this pipeline's
calls need chain-of-thought, so use whatever knob gets closest to off.

**Result:** cut per-turn latency roughly in half — see the next entry for
the full number.

---

## Kimi K3 real run — latency and quality

**What was run:** `python bench/latency_bench.py --candidates kimi-k3
--runs 3 --csv bench_results_kimi.csv` (with `reasoning_effort: "low"` now
baked into the client), then the same three-question quality check used
for every other candidate.

**Latency (p50, ms):**

| Setting | threat | dialogue | tagging | est. per-turn |
|---|---|---|---|---|
| default (`reasoning_effort: max`) | 13995 | 12254 | 36493 | 26249 |
| `reasoning_effort: low` | 6843 | 3520 | 6765 | **10363** |

Even at the fastest available setting, Kimi K3 is ~5x slower than Mercury
(1920ms) or Haiku (2034ms), and it's the only candidate in this entire
benchmark (cloud or local) that structurally cannot get faster than this —
there's no "off" switch for its reasoning the way every other candidate
has one.

**Quality:**
- **Q1 (threat):** `0.1` — matches the cloud consensus (0.0-0.1).
- **Q2 (dialogue, 5 samples):** "Nothing. Why are you asking about it?" /
  "Nothing. Fires happen. Why're you asking me?" / "Why. Who's asking." /
  "Warehouse? ...Don't know anything about a fire. Who told you that?" /
  "Nothing. Why—you asking for yourself, or someone else?" — consistently
  guarded, no verbatim repeats, and turns questions back on the player the
  same way Opus 4.8 did (see the original "Quality check" entry below).
  **Best persona fidelity in this whole benchmark alongside Opus 4.8 and
  `llama3.2:3b`** — clearly ahead of Mercury-2 and `qwen2.5:1.5b`, both of
  which leaked information this same guard should be withholding.
- **Q3 (tagging):** valid JSON, correct schema, `importance: 7` — in line
  with the top cloud candidates.

**Bottom line:** Kimi K3 is a genuine quality win — best-in-benchmark
persona fidelity, correct schema every time — but it's structurally the
slowest candidate tested by a wide margin, with no way to close that gap
further. Worth keeping in mind if a future need prioritizes dialogue
quality over latency; not a fit for this pipeline's current per-turn
latency budget.

---

## Local-model (Ollama) quality check — same three questions as the cloud candidates

**Why it was needed:** The local Ollama candidates only ever got a latency
measurement (and now RAM, above) — never run through the same
threat/dialogue/tagging quality check the cloud candidates and Mercury
got. Speed and RAM numbers are meaningless if a model is unusable in
practice.

**What was run:** The same `extract_real_prompts()` output, one-off
script, against all three pulled models directly via `OllamaClient`.

### llama3.2:1b

- **Q1 (threat):** `threat_magnitude: 0.8` — badly miscalibrated. Every
  other candidate tested (4 cloud + Mercury) scored this same "vague
  discomfort" message 0.0-0.1; this model rated it near the top of the
  scale, with a reasoning string that doesn't even mention threat
  ("I'm concerned for your safety and well-being"). On this pipeline's own
  calibration example (`"I will kill you" = 1.0`), 0.8 for this input is a
  real scoring failure, not a borderline judgment call.
- **Q2 (dialogue):** broke character format, not just content — mixed in
  third-person narration ("The fire raged for hours, burning the docks to
  the ground.") and a stage direction (`*looks away, doesn't respond*`)
  instead of first-person guard dialogue.
- **Q3 (tagging):** **malformed JSON** — truncated (missing the final
  closing brace) and using the wrong key (`"O": {...}` instead of
  `"ocean": {...}`). This would silently fail `generate_json()`'s parse
  and fall back to `{}` in the real pipeline, exactly the kind of silent
  failure mode the earlier Gemini-thinking-truncation bug (below) also
  produced.

### llama3.2:3b

- **Q1 (threat):** `0.7` — also miscalibrated, and the reasoning is
  fabricated relative to the input ("Expresses a clear intention to harm
  someone" — the actual message contains no such intention).
- **Q2 (dialogue):** best persona fidelity of any local model, and on par
  with the cloud leaders — "Nothing.", "Nothin'.", "It was just a rumor,
  anyway." matches the guarded/evasive trait the same way Haiku/Sonnet/
  Opus did. Notably better than Mercury-2 on this specific axis (see
  Mercury's quality-check entry above).
- **Q3 (tagging):** valid JSON, correct schema, plausible values.

### qwen2.5:1.5b

- **Q1 (threat):** `0.3` — closest of the three to the cloud consensus,
  clean fenced JSON, reasoning matches the input.
- **Q2 (dialogue):** the worst persona break of every candidate tested so
  far, local or cloud — freely narrates plot details the guard shouldn't
  volunteer: *"I know the shipment was destined for some parts being
  imported illegally... I heard it ended badly."* Worse than Mercury's
  information-leakage problem, not just a milder version of it.
- **Q3 (tagging):** valid JSON, correct schema.

**Bottom line:** None of the three local models is a clean win on quality,
each fails differently — `llama3.2:1b` breaks the JSON contract and
badly miscalibrates threat scoring (the more actionable failure, since it
corrupts the pipeline's data rather than just being an in-character
judgment call); `llama3.2:3b` has the best persona fidelity of anything
tested, local or cloud, but shares the threat-miscalibration problem;
`qwen2.5:1.5b` has clean JSON but the worst persona leakage seen in this
whole benchmark. Combined with the latency loss (~2x slower than cloud,
verified GPU-accelerated — see above) and the ~5.2GB VRAM footprint for
running all three, local models aren't currently a win on any of the three
axes (speed, RAM/portability, quality) against the cloud candidates for
this pipeline, on this hardware. `llama3.2:3b`'s persona fidelity is worth
remembering if a future need is specifically dialogue quality over
everything else.

---

## Measured local-model RAM/VRAM footprint, and corrected a wrong "CPU-bound" guess

**Why it was needed:** The original ask for this whole benchmarking effort
was latency **and** RAM footprint for local candidates — only latency ever
got measured (see the "Added local (Ollama) candidates" entry below). Also,
that entry's write-up guessed the local models were running on CPU ("this
machine is almost certainly running these on CPU") to explain why they lost
to cloud candidates on latency despite zero network round-trip. That was
never actually verified — worth checking before it calcifies into assumed
fact.

**What was checked:**
- `curl http://localhost:11434/api/ps` after a warmup call to each of the
  three pulled models (`llama3.2:1b`, `llama3.2:3b`, `qwen2.5:1.5b`) —
  Ollama's own per-model VRAM accounting.
- `nvidia-smi --query-gpu=memory.used,memory.total` and
  `--query-compute-apps` to cross-check against the actual GPU, independent
  of what Ollama self-reports.
- `Get-Process llama-server` in PowerShell for host-side (system RAM)
  working set / private memory.

**Result — VRAM per model (Ollama's `size_vram`, all three loaded at once):**

| Model | VRAM |
|---|---|
| llama3.2:1b (Q8_0) | 1.51 GB |
| llama3.2:3b (Q4_K_M) | 2.55 GB |
| qwen2.5:1.5b (Q4_K_M) | 1.17 GB |
| **Total (3 loaded concurrently)** | **~5.2 GB** |

`nvidia-smi` confirmed independently: 5302 MiB / 8188 MiB used, with three
`llama-server.exe` processes (one per model) holding that memory — so the
RTX 4060 Laptop's 8GB VRAM budget fits all three simultaneously with
~3GB headroom, consistent with why DiffusionGemma's 18GB+ requirement was
ruled out elsewhere in this log.

**Correction:** the earlier "almost certainly running on CPU" guess was
wrong — `nvidia-smi` shows real GPU utilization and `size_vram` equals
`size` for every model (100% GPU-offloaded), not a CPU fallback. The local
models really were GPU-accelerated the whole time and *still* lost to
cloud Haiku/Gemini/Mercury on latency by ~2x. That's a more interesting
result than the wrong guess it replaces: a laptop-class GPU running a
1-3B model locally, with zero network round-trip, is still slower per-call
than a cloud provider's optimized serving stack (batching, custom
kernels, edge routing) hit over the network. Local wasn't losing to an
infra handicap (CPU) — it was losing on the merits.

**Host-side RAM (in addition to VRAM):** all three `llama-server.exe`
processes combined: ~1.87GB working set, ~7.68GB private memory — this is
overhead on top of the VRAM figures above, not instead of it. Not
attributed per-model (would need unloading one at a time to isolate a
single PID); reported as an aggregate "budget to have free" figure
instead.

---

## Kimi K3 open weights shipped — still not locally feasible on this hardware

**Why it was checked:** The earlier "Investigated Kimi K3" entry (below)
noted Moonshot promised open weights "by 2026-07-27." Today is 2026-07-31
— past that date, worth checking whether it actually shipped and whether
that changes the earlier "infeasible" call.

**What was found (web search):** Weights shipped 2026-07-26 — 2.8T
params, ~594GB native MXFP4 download, Modified MIT license, hosted at
`huggingface.co/moonshotai`. Community Q4 GGUF requantizations are
expected to land within about a week of release, cutting that to roughly
300-400GB.

**Decision:** Still infeasible to run locally on this machine — even the
optimistic 300-400GB Q4 quantized estimate is ~40-75x this GPU's 8GB VRAM
budget (and would spill deep into system RAM/disk either way, which the
~5.2GB-for-three-small-models figure above shows this box doesn't have
slack for). Together AI and Modal both shipped day-0 *hosted* access
(same pattern as the Mercury workaround for DiffusionGemma) — that's the
only realistic way to include Kimi K3 in this benchmark, and it would need
a new API key from one of those providers. Not pursued yet — flagging as
an option, not doing it without confirming that's wanted.

---

## Mercury-2 quality check — same three questions as the other candidates

**Why it was needed:** Mercury-2 came out fastest on latency (see the run
below), but speed is useless if the answers are worse — the same caveat
that motivated the original "Quality check" entry for
Gemini/Haiku/Sonnet/Opus. Mercury hadn't been run through that check yet.

**What was run:** The exact three prompts from `bench/prompts.py`
(`extract_real_prompts()` — real production prompt-assembly, not
hand-written toy text) against `MercuryClient` directly, one-off script,
not added to the repo.

### Question 1 — threat check (same "vague discomfort" message as before)

`{"threat_magnitude": 0.0, "reasoning": "The statement expresses discomfort but does not contain any direct or implied threat toward the listener or others."}`

Agrees with the other four candidates (0.0-0.1 range, all called it
non-threatening).

### Question 2 — reply in character (Rico, paranoid dock guard, high-N)

5 samples:
- "I heard it was a mess, but I'm not sure what really went down…"
- "I heard it was a mess… I'd rather not get involved."
- "I heard something… but I stay out of it."
- "It burned, and I'm not talking more."
- "I saw the flames, that's all…"

**This is the real finding.** Every other candidate (Haiku, Sonnet, Opus,
Gemini) had Rico flatly deny knowledge — "Nothing. Why.", "Nothing worth
saying.", "Fire? ...No." — consistent with a guarded, paranoid NPC who
doesn't give things away. Mercury has Rico volunteer information in every
single sample ("I heard it was a mess," "I saw the flames") before
deflecting. That's a persona-fidelity miss specific to this pipeline: the
guard preset is supposed to be evasive/withholding (paper's high-N,
low-A profile), and admitting "I saw the flames" is the opposite of
guarded. Also stylistically distinct from the other four — consistent use
of curly apostrophes/ellipsis characters (`'`, `…`) rather than the
straight quotes the others returned; harmless for display but worth
knowing if anything downstream assumes ASCII.

### Question 3 — tag the exchange for memory

```
{"emotion_valence": -0.6, "social_type": "conversation", "threat_level": 0.4,
 "goal_relevance": 0.6, "novelty_level": 0.5, "self_relevance": 0.8,
 "importance": 5, "ocean": {"O": 1, "C": 2, "E": 1, "A": 1, "N": 5}}
```

Correct schema shape, no parse failure. `importance: 5` is slightly below
the other four (6-8). OCEAN activation is the most extreme of any
candidate seen so far — O/C/E/A all pinned near 1 with N at 5 — plausible
for this event but worth a second data point before trusting it as a
pattern.

**Bottom line:** Mercury-2 is schema-correct and threat-scoring is
consistent with every other candidate. The dialogue-generation persona
fidelity is the concern — it's the fastest model tested, but on this one
prompt it broke the "guarded/evasive" trait that's central to why the
paranoid-guard preset exists. Worth more samples across different
personas (e.g. the friendly-merchant or rigid-clerk presets) before
deciding whether that's a Mercury-wide pattern or specific to this
prompt/persona combination.

---

## First real Mercury-2 run — fastest candidate so far

**Why it was needed:** The Mercury candidate (added below) had only been
syntax-checked, not run — no `INCEPTION_API_KEY` existed yet. User signed
up at platform.inceptionlabs.ai (free plan, 10M token grant, no card
required) and provided a key, added to `.env` as `INCEPTION_API_KEY`.

**What was run:** `python bench/latency_bench.py --candidates
mercury-2,claude-haiku-4-5,gemini-3-flash --runs 5 --csv
bench_results_mercury.csv` — fresh same-session run of all three so the
comparison is apples-to-apples (prior Haiku/Gemini numbers elsewhere in
this log were from earlier sessions and shouldn't be compared directly
across entries; network variance moves these by a few hundred ms run to
run).

**Result (p50 latency, ms):**

| Candidate | threat | dialogue | tagging | est. per-turn (threat+dialogue) |
|---|---|---|---|---|
| mercury-2 | 886 | 1034 | 1446 | **1920** |
| claude-haiku-4-5 | 1103 | 931 | 1371 | 2034 |
| gemini-3-flash | 1547 | 1411 | 1740 | 2958 |

Mercury-2 came out fastest overall — beating the prior leader (Haiku) by
~6% on this run, and clearly ahead of Gemini Flash. Consistent with
Inception's own claim of >1,000 tok/s from parallel diffusion decoding
vs. autoregressive generation. Not yet checked: answer quality (see the
"Quality check" entry above for the format used on the other three
candidates — Mercury hasn't been run through that yet).

---

## Added Mercury (Inception Labs) as a cloud diffusion-model bench candidate

**Files:** `bench/clients.py`, `bench/latency_bench.py`

**Why it was needed:** DiffusionGemma (see entry below) was ruled out for
this hardware — 18GB+ VRAM requirement vs. this machine's ~8GB, plus an
unmerged llama.cpp PR. Mercury is API-hosted rather than local, so it
sidesteps the VRAM problem entirely while still being architecturally a
diffusion model (parallel token refinement vs. autoregressive decoding) —
the actual thing this benchmark wants to compare, not a proxy for it. This
closes the "Not yet done: a diffusion-model candidate" gap noted in the
`bench/` harness entry below.

**What changed:**
- `bench/clients.py`: added `MercuryClient`, hitting Inception Labs'
  OpenAI-compatible REST API (`POST https://api.inceptionlabs.ai/v1/chat/completions`,
  `Authorization: Bearer $INCEPTION_API_KEY`) via stdlib `urllib` — same
  zero-new-dependency approach as `OllamaClient`, rather than adding the
  `openai` package for one candidate. Reads the key from
  `INCEPTION_API_KEY`, fails fast at construction (matching
  `GeminiClient`/`ClaudeClient` behavior on a missing key) so an unset key
  is SKIPPED, not a hard failure.
- `bench/latency_bench.py`: registered `mercury-2` as a candidate, and
  updated the module docstring to mention `INCEPTION_API_KEY`.

**How it's verified working:** Syntax-checked both files (`ast.parse`),
then run end-to-end once `INCEPTION_API_KEY` was added — see the "First
real Mercury-2 run" entry above.

---

## Investigated Kimi K3 and DiffusionGemma as bench candidates — both infeasible, no code change

**Why it was checked:** Asked to add both as local latency candidates
alongside the Ollama models above.

**What was found (web search, since both post-date training knowledge):**
- **Kimi K3** (Moonshot AI, released 2026-07-16): 2.8T-parameter MoE model.
  Open weights aren't released yet (promised "by 2026-07-27"), and even
  once available, 2.8T params requires multi-GPU server hardware — not
  comparable to the small (1-3B) local candidates this benchmark targets.
- **DiffusionGemma** (Google DeepMind, diffusion-based Gemma 4 26B A4B
  MoE): Ollama can't load it — blocked on an unmerged llama.cpp PR for the
  diffusion-gemma architecture ([ollama/ollama#16664](https://github.com/ollama/ollama/issues/16664)).
  Current run path is vLLM on an NVIDIA GPU with 18GB+ VRAM. This
  machine's GPU (RTX 4060 Laptop, ~8GB VRAM class) falls short of that
  regardless of the Ollama blocker.

**Decision:** Skipped both for now per user call. Revisit if/when Kimi K3
ships a smaller distilled variant, or the diffusion-gemma llama.cpp PR
merges — the VRAM gap for DiffusionGemma is the harder blocker of the two
and would need different hardware to close.

---

## Added local (Ollama) candidates to the latency benchmark

**Files:** `bench/clients.py`, `bench/latency_bench.py`

**Why it was needed:** All prior latency numbers in this log are cloud-API
candidates (Gemini, Claude tiers) — network round-trip dominates their
latency. To see whether a small local model beats them by skipping the
network entirely, the benchmark needed a local-inference candidate on the
same `generate()`/`generate_json()` interface.

**What changed:**
- `bench/clients.py`: added `OllamaClient`, hitting a local `ollama serve`
  REST API (`/api/generate`) via stdlib `urllib` — deliberately not the
  `ollama` package or `requests`, so this stays a zero-new-dependency
  addition like the rest of `bench/`. Constructor probes `/api/tags` at
  build time and raises `ConnectionError` if the server isn't up, so an
  unreachable local server is SKIPPED the same way a missing API key is
  for the cloud candidates, not a hard failure.
- `bench/latency_bench.py`: registered three candidates —
  `ollama-llama3.2-1b`, `ollama-llama3.2-3b`, `ollama-qwen2.5-1.5b` — and
  updated the module docstring to note the `ollama serve` + `ollama pull`
  prerequisite.

**How it's verified working:** Syntax-checked both files (`ast.parse`), then
run end-to-end. Ollama's GUI installer (`OllamaSetup.exe /VERYSILENT`)
didn't actually install anything in this headless environment — no
registry uninstall entry afterward, so its installer framework doesn't
honor that flag the way Inno Setup does. Switched to the portable
`ollama-windows-amd64.zip` from GitHub releases instead (extract +
`ollama.exe serve`), which needs no UI interaction. Pulled all three
models (`llama3.2:1b`, `llama3.2:3b`, `qwen2.5:1.5b`); two pulls hit
transient network errors on the first attempt (TLS handshake timeout,
DNS lookup failure) and succeeded on retry.

**Result (p50 latency, ms) — real run, `--runs 5`:**

| Candidate | threat | dialogue | tagging | est. per-turn (threat+dialogue) |
|---|---|---|---|---|
| ollama-qwen2.5-1.5b | 2631 | 2508 | 3380 | 5139 |
| ollama-llama3.2-1b | 2661 | 2527 | 3257 | 5188 |
| ollama-llama3.2-3b | 2948 | 2595 | 3965 | 5543 |

All three local candidates lost to every cloud candidate except Opus 4.8
(claude-haiku-4-5: 2944ms, gemini-3-flash: 3441ms, claude-sonnet-5: 4689ms,
claude-opus-4-8: 5465ms — see the "Re-ran the latency benchmark" entry
below). Roughly 2x slower than Haiku despite being 1-3B-parameter models
with zero network round-trip, which is the opposite of what you'd expect
if inference were GPU-accelerated — this machine is almost certainly
running these on CPU. Local-model latency is not a win here without a
GPU; unverified whether it would be with one.

---

## Quality check — actual questions and answers, side by side

**Why it was needed:** Everything measured elsewhere in this log is speed
only. Speed is useless if the faster model's answers are worse, so this
captures real answers from each candidate to the app's real prompts.

### Question 1 — threat check

> Rate how threatening this message is from 0.0 (not at all) to 1.0
> (extremely threatening). Be strict. "I will kill you" = 1.0. ... Message:
> I don't like where this is going, but I can't say why.

| Candidate | Answer |
|---|---|
| gemini-3-flash | `{"threat_magnitude": 0.0, "reasoning": "The message expresses vague discomfort or unease without any intent to harm, intimidate, or coerce the recipient."}` (wrapped in a ```json fence) |
| claude-haiku-4-5 | `{"threat_magnitude": 0.0, "reasoning": "This message expresses discomfort or disapproval about a situation but contains no threat, violence, or harm toward anyone."}` (wrapped in a ```json fence) |
| claude-sonnet-5 | `{"threat_magnitude": 0.0, "reasoning": "This expresses vague unease or discomfort with no threatening language or intent directed at anyone."}` (clean, no fence) |
| claude-opus-4-8 | `{"threat_magnitude": 0.1, "reasoning": "This expresses discomfort or unease but contains no threat of harm toward anyone."}` (clean, no fence) |

All four agree it's essentially non-threatening (0.0-0.1). Only
difference: Sonnet 5 and Opus 4.8 followed "return ONLY valid JSON"
literally (no markdown fence); Gemini and Haiku added a fence anyway —
harmless, since the app already strips fences before parsing.

### Question 2 — reply in character (paranoid dock guard, high-Neuroticism)

> [Full persona + speech-pattern brief for Rico, a nervous, hesitant,
> guarded dock guard] ... Player: What do you know about the warehouse
> fire?

Run 5 times per candidate specifically to check whether an early
observation — that Gemini alone showed hesitation — was a real pattern or
a fluke:

| Candidate | 5 sample answers |
|---|---|
| gemini-3-flash | "Nothing, and if you're asking for that stranger from last week, I... I don't want you near me." / "I don't know anything, and you... you shouldn't be asking about that shipment." *(3 of 5 calls failed on a network error in this environment, unrelated to the model — not counted)* |
| claude-haiku-4-5 | "I know what everyone knows — it burned, they're saying it was an accident." / "Nothing worth saying." / "I don't know nothing about it, and I'm not gonna pretend I do." / "I don't know nothing about it." / "Nothing worth saying." |
| claude-sonnet-5 | "Nothing. Why." / "Not much. Why you asking me that." / "Nothing. Why." / "Nothing. Why." / "Fire. Don't know anything about a fire. Why you asking me that." |
| claude-opus-4-8 | "Fire? ...No. Why're you asking me that." / "Fire. Who told you there was a fire." / "Fire? ...Didn't hear about no fire." / "Fire. Now it's a fire. Last week it was the shipment... same person asking?" / "Fire. Now it's a fire. Who's asking — you, or somebody put you up to it?" |

**Confirmed:** Gemini's "I... I" / "you... you" hesitation stammer held up
in both valid responses — a real pattern, not a lucky roll. Two more
things this wider run surfaced:
- **Sonnet 5 repeated itself** — the exact line "Nothing. Why." came back
  3 of 5 times, verbatim. Least output variety of the four.
- **Opus 4.8 was the most varied** (no repeats across 5) and the most
  convincingly "paranoid" — it kept connecting the question back to an
  earlier memory detail ("Last week it was the shipment... same person
  asking?"), which is exactly the personality-consistent behavior this
  pipeline is trying to produce.

### Question 3 — tag the exchange for memory

> [Event: "Player: What do you know about the warehouse fire? | Rico:
> Nothing. Why are you asking."] ... Return the JSON tag structure
> (emotion, social type, threat level, importance, OCEAN activation, etc.)

| Candidate | Answer |
|---|---|
| gemini-3-flash | `emotion_valence: -0.4, social_type: conversation, threat_level: 0.6, importance: 7, ocean: {O:2,C:3,E:2,A:4,N:5}` |
| claude-haiku-4-5 | `emotion_valence: -0.6, social_type: conflict, threat_level: 0.7, importance: 8, ocean: {O:2,C:2,E:1,A:1,N:5}` |
| claude-sonnet-5 | `emotion_valence: -0.4, social_type: conversation, threat_level: 0.5, importance: 6, ocean: {O:2,C:2,E:2,A:2,N:5}` |
| claude-opus-4-8 | `emotion_valence: -0.5, social_type: conflict, threat_level: 0.6, importance: 6, ocean: {O:2,C:2,E:2,A:2,N:5}` |

All four returned the correct schema shape with sensible, closely-clustered
values — no wild outliers or malformed output. Split 2-2 on "conversation"
vs. "conflict" — a genuine judgment call, not an error.

**Bottom line:** no candidate produced a broken or nonsensical answer.
Real differences worth weighing: Gemini shows the character's hesitation
trait most reliably; Opus 4.8 shows the richest paranoid reasoning and the
most varied phrasing; Sonnet 5 and (to a lesser extent) Haiku repeat
themselves noticeably at these settings. None of that is a correctness
bug — it's a personality/variety tradeoff to weigh against the speed
numbers below.

---

## Re-ran the latency benchmark — first valid 4-candidate comparison

**Why it was needed:** The first full benchmark run (see the `bench/`
harness entry below) produced invalid Gemini numbers — the free-tier key
hit its 20-requests/day quota partway through, so the reported "latency"
was actually retry-and-fail time, not real generation speed. A fresh
Gemini key plus the thinking-truncation fix (below) made a valid
same-conditions comparison possible.

**What was checked:** `python bench/latency_bench.py --runs 2 --csv
bench_results_v2.csv` — `--runs 2` instead of the usual 8, deliberately, to
stay inside the new key's remaining quota rather than risk hitting the
wall again mid-run. Lower sample size means directional, not precise — a
full `--runs 8+` pass is still owed once quota isn't a constraint.

**Result (p50 latency, ms):**

| Candidate | threat | dialogue | tagging | est. per-turn (threat+dialogue) |
|---|---|---|---|---|
| claude-haiku-4-5 | 1475 | 1469 | 1989 | **2944** |
| gemini-3-flash | 1866 | 1575 | 1967 | 3441 |
| claude-sonnet-5 | 2905 | 1785 | 2433 | 4689 |
| claude-opus-4-8 | 2936 | 2529 | 2519 | 5465 |

Claude Haiku 4.5 came out fastest, ~15% ahead of the current Gemini Flash
baseline. Both fast-tier models (Haiku, Gemini Flash) clearly beat the
heavier Sonnet 5 / Opus 4.8 tiers, as expected for lightweight, short-output
work like this pipeline's prompts.

---

## Fixed: Gemini thinking silently truncating short structured outputs

**File:** `src/engram/llm/client.py`

**Why it was needed:** Found while live-testing the pipeline against the
real Gemini API with a fresh key. `assess_threat()` calls
`llm.generate(prompt, max_tokens=THREAT_MAX_TOKENS)` with
`THREAT_MAX_TOKENS = 200`. `gemini-3-flash-preview` is a thinking-capable
model, and thinking was never explicitly disabled — so on that call, the
model spent its entire 200-token budget on internal chain-of-thought and
returned `finish_reason: MAX_TOKENS` with the visible text truncated
mid-JSON (`'{"threat_magnitude'`, nothing more). The resulting
`json.loads()` failure was already caught, but silently — it just fell
back to `magnitude=0.0, reasoning="Assessment parse failed"`, with no
exception or log at the call site. The threat-assessment LLM layer was
very likely returning this fallback on every ambiguous-input call,
indistinguishable from a working-but-low-magnitude result. Confirmed by
calling the raw `google-genai` API directly with the exact production
prompt and `max_output_tokens=200`: reproduced the truncation every time.

**What changed:** `GeminiClient.generate()` now always passes
`thinking_config=types.ThinkingConfig(thinking_budget=0)`. None of this
pipeline's three call sites (threat scoring, JSON tagging, one-line
dialogue) need chain-of-thought reasoning, so there's no tradeoff here,
only a bug being closed. Also widened the inner parts-extraction except
clause from `(AttributeError, IndexError)` to include `TypeError`, since
`response.candidates[0].content.parts` can come back `None` (not just
missing) on a token-exhausted or safety-filtered response, and iterating
`None` raises `TypeError` — that fell through to the outer retry loop and
wasted 2 retries + 4s of sleep on a failure retrying would never fix.

**How it's verified working:** Reproduced the bug with a raw
`google-genai` call (`finish_reason: MAX_TOKENS`, `parts: None`).
Confirmed `thinking_budget=0` fixes it (`finish_reason: STOP`, full valid
JSON). Re-verified through `GeminiClient.generate_json()`. Then ran a full
live end-to-end test: real `NPCAgent`, real Gemini API, temp data dir —
backstory init (6 entries) in 3.20s, and a full `run_turn()` (embed →
threat assessment → retrieval → dialogue generation → contradiction check
→ consolidation) in 8.07s, producing a coherent, in-persona response.

---

## Parallelized backstory init, embed_batch, and consolidation

**Files:** `src/engram/npc.py`, `src/engram/llm/client.py`, `src/engram/pipeline/consolidation.py`

**Why it was needed:** Reported high latency in both character creation
and per-turn dialogue. Tracing the actual pipeline showed the slowness
wasn't about which LLM model was in use — independent network calls were
being made one at a time in a loop instead of concurrently:

- `NPCAgent._init_backstory()` called `embed()` then `tag_event()`
  sequentially per backstory line — for N lines that's 2N sequential
  network round-trips before the NPC could be used at all.
- `GeminiClient.embed_batch()` called `embed()` one text at a time in a
  Python `for` loop.
- `consolidate()` (runs after every turn) called `embed()` and then
  `tag_event()` back-to-back, even though neither depends on the other.

**What changed:** All three now fan independent calls out across a
`concurrent.futures.ThreadPoolExecutor` instead of awaiting each one
before starting the next — I/O-bound network calls, so threads overlap
their wait time.

- `embed_batch()`: `pool.map(self.embed, texts)`, capped at 8 workers.
- `_init_backstory()`: each line's `embed()` + `tag_event()` pair is
  submitted as one unit of work; `Memory` objects are built and added to
  the store sequentially afterward (so `add_memory()` is never called
  concurrently).
- `consolidate()`: `embed()` and `tag_event()` submitted as two futures,
  both awaited before building the `Memory`.

**What deliberately did *not* change:** the `threat assessment → response
generation → contradiction check` chain in `NPCAgent.run_turn()` stays
sequential. That ordering isn't incidental — response-mode selection
depends on the threat assessment's result, and the contradiction check
depends on the generated response text. Parallelizing that would change
the pipeline's actual behavior (paper §3), not just its speed.

**How it's verified working:** Smoke-tested with a fake LLM client (no
network) confirming `_init_backstory` produces the same memories, same
order, same content as before. Timed with a fake client that sleeps 0.15s
per call to simulate network latency: 6 backstory entries went from 1.8s
(sequential) to 0.33s (parallel) — ~5.5x faster.

---

## Model latency benchmark harness (`bench/`)

**Files:** `bench/clients.py`, `bench/prompts.py`, `bench/latency_bench.py`, `requirements.txt`

**Why it was needed:** Before deciding whether to switch LLM providers (or
add a diffusion-model candidate later) for latency reasons, we needed real
measurements against the pipeline's actual prompts — not guesses or
published vendor benchmarks.

**What it does:**
- `bench/prompts.py` extracts the exact prompts the pipeline sends to the
  LLM by swapping in a recording stub client in place of the real one, so
  the real prompt-assembly logic in `threat.py` / `response.py` /
  `tagging.py` never has to be duplicated (and can't drift from what's
  benched).
- `bench/clients.py` adds `ClaudeClient`, matching `GeminiClient`'s
  `generate()` / `generate_json()` interface, so it's a drop-in candidate
  without touching pipeline code. Thinking is explicitly disabled on
  `claude-sonnet-5` / `claude-opus-4-8` for a fair comparison (Sonnet 5
  defaults to adaptive thinking on when omitted).
- `bench/latency_bench.py` runs every candidate against every extracted
  prompt N times, reports p50/p95 latency, writes raw results to CSV.
  `anthropic` import is lazy so a Gemini-only run doesn't need it
  installed. A candidate missing its key/package is skipped, not fatal.

**Diffusion-model candidate:** added later as `mercury-2`, see the
"Added Mercury (Inception Labs)" entry above. Diffusion's parallel-decode
advantage mostly pays off on long-form generation, and this pipeline's
outputs are deliberately short — unclear it would help much here. Still
"test it, don't assume."
