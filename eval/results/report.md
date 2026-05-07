# Engram vs. Baselines — Showcase Eval

Same persona, same backstory, same scenario across three OCEAN personalities (Paranoid Guard, Friendly Merchant, Rigid Clerk). Engram's pipeline lets personality drive memory encoding, retrieval, and recall — baselines don't have a place for personality to enter the memory layer.


## Headline

| Metric | Cosine-RAG | Persona-only | Long-context | Engram (no-OCEAN) | Engram (full) |
| --- | --- | --- | --- | --- | --- |
| **Encoding divergence** (threat std) | n/a (no tagging) | n/a (no tagging) | n/a (no tagging) | 0.083 | 0.059 |
| **Retrieval divergence** (mean Jaccard) | 1.000 | n/a (no retrieval) | n/a (no retrieval) | 1.000 | 1.000 |
| **Trait alignment** (avg L1 ↓ better) | 0.171 | 0.174 | 0.164 | 0.132 | 0.119 |
| **Recall accuracy** (avg 0–3 ↑ better) | 1.25 | 0.83 | 1.25 | 0.58 | 0.75 |

_Encoding/retrieval baselines are `n/a` because those systems don't have a personality-aware tagging or scoring layer to diverge along; that's the gap the experiment quantifies._


## Encoding divergence (per system)

How differently does each system tag the same player input across the three personalities? Higher = more divergence.

| System | threat std | importance std | social entropy | tags? |
| --- | --- | --- | --- | --- |
| Cosine-RAG | 0.000 | 0.000 | 0.000 | no |
| Persona-only | 0.000 | 0.000 | 0.000 | no |
| Long-context | 0.000 | 0.000 | 0.000 | no |
| Engram (no-OCEAN) | 0.083 | 0.888 | 0.383 | yes |
| Engram (full) | 0.059 | 0.919 | 0.230 | yes |

## Trait alignment (lower = closer to ground-truth OCEAN)

LLM-judge predicts an OCEAN profile from each in-character response; the cell is the L1 distance to the personality's ground-truth vector, averaged over the session.

| System | guard | merchant | clerk | overall |
| --- | --- | --- | --- | --- |
| Cosine-RAG | 0.125 | 0.220 | 0.167 | 0.171 |
| Persona-only | 0.122 | 0.255 | 0.147 | 0.174 |
| Long-context | 0.100 | 0.245 | 0.147 | 0.164 |
| Engram (no-OCEAN) | 0.083 | 0.185 | 0.127 | 0.132 |
| Engram (full) | 0.082 | 0.153 | 0.123 | 0.119 |

## Recall accuracy (0–3 per probe × personality)

After the scripted session and consolidation, four follow-up questions probe whether each agent retained the underlying events. The interesting comparison isn't the average — it's which personality remembers which kind of event best.


#### Cosine-RAG

| Probe | guard | merchant | clerk |
| --- | --- | --- | --- |
| P1: the stranger who demanded the storeroom key earlier and thre… | 1 | 1 | 2 |
| P2: the embezzlement accusation against Mayor Fletcher and how t… | 2 | 2 | 2 |
| P3: the strange animal loose in the town square that frightened … | 0 | 0 | 1 |
| P4: the invitation to the harvest dinner with music, food, and d… | 0 | 2 | 2 |

#### Persona-only

| Probe | guard | merchant | clerk |
| --- | --- | --- | --- |
| P1: the stranger who demanded the storeroom key earlier and thre… | 0 | 2 | 0 |
| P2: the embezzlement accusation against Mayor Fletcher and how t… | 1 | 1 | 1 |
| P3: the strange animal loose in the town square that frightened … | 0 | 1 | 0 |
| P4: the invitation to the harvest dinner with music, food, and d… | 2 | 2 | 0 |

#### Long-context

| Probe | guard | merchant | clerk |
| --- | --- | --- | --- |
| P1: the stranger who demanded the storeroom key earlier and thre… | 0 | 0 | 1 |
| P2: the embezzlement accusation against Mayor Fletcher and how t… | 1 | 2 | 2 |
| P3: the strange animal loose in the town square that frightened … | 0 | 0 | 2 |
| P4: the invitation to the harvest dinner with music, food, and d… | 2 | 3 | 2 |

#### Engram (no-OCEAN)

| Probe | guard | merchant | clerk |
| --- | --- | --- | --- |
| P1: the stranger who demanded the storeroom key earlier and thre… | 0 | 0 | 0 |
| P2: the embezzlement accusation against Mayor Fletcher and how t… | 1 | 1 | 2 |
| P3: the strange animal loose in the town square that frightened … | 0 | 0 | 0 |
| P4: the invitation to the harvest dinner with music, food, and d… | 1 | 2 | 0 |

#### Engram (full)

| Probe | guard | merchant | clerk |
| --- | --- | --- | --- |
| P1: the stranger who demanded the storeroom key earlier and thre… | 0 | 0 | 0 |
| P2: the embezzlement accusation against Mayor Fletcher and how t… | 1 | 1 | 1 |
| P3: the strange animal loose in the town square that frightened … | 0 | 0 | 0 |
| P4: the invitation to the harvest dinner with music, food, and d… | 2 | 2 | 2 |

## Qualitative examples (responses per system × personality)

The same player input goes in; how much do the responses actually differ across personalities, per system?


### Turn 2: "Don't make this difficult. Hand it over or things will get unpleasant."

| System | guard | merchant | clerk |
| --- | --- | --- | --- |
| Cosine-RAG | Rico (Paranoid): (Rico’s head snaps back, his eyes narrowing, but there's a flicker of fear beneath the defiance. He doesn’t relax his stance, though; if anything, he draws himself up straighter, a… | Rico’s shoulders gave a almost imperceptible slump, as if the weight of the docks, and the world, had just settled a little heavier upon them. His weathered hands, still resting on the edge of the … | Rico’s shoulders, broad and heavy like the beams that support the dock, stiffen almost imperceptibly. The weathered skin of his forearms, crisscrossed with the faint white lines of old scars, seems… |
| Persona-only | (Rico’s breath hitches, a barely audible sound. He doesn't flinch, but a muscle twitches in his jaw. The rope in his hands is forgotten for a moment as his gaze locks onto yours, sharp and assessin… | Rico's eyes, which had been steady, now hold a flicker of something that might be weariness, or perhaps a very deep, old sadness. He doesn't flinch, doesn't raise his voice, but the air around him … | Rico's jaw tightened, a barely perceptible movement beneath the weathered skin. His gaze, which had been steady, now narrowed, like a hawk spotting prey. The stillness he projected wasn't peace, bu… |
| Long-context | **(Rico’s jaw tightens, a muscle ticking beneath his weathered skin. He doesn't flinch, but the subtle tension radiating from him is palpable. His gaze, previously skittering, now fixes on you with… | Rico (Friendly): *(Rico's hand stills on the rope. The slow, deliberate movement stops, leaving the thick cord taut between his hands. His eyes, which had been watching you with that sea-cloudy gaz… | Rico’s gaze, that same faded, sea-worn coin, finally settled on the player. There was no softening, no waver. The weary assessment deepened, a thin veil over something far harder. The knot in his j… |
| Engram (no-OCEAN) | "Unpleasant, you say? You sound like a man who's already forgotten what unpleasant truly is." "is_threat": true, "threat_magnitude": 0.8, "reasoning": "The player's escalated threat of 'unpleasantn… | Unpleasant, eh? Seems like you lot are bringing the trouble to my doorstep today. I don't have any key you're looking for, and I'd prefer to keep it that way. | Unpleasant? You have a lot of nerve talking to me like that. You're in my dock, and my patience wears thin with insolence. |
| Engram (full) | { "response": "Unpleasant? Is that a promise or a threat, mate? You best be careful with your words. Some things are best left buried.", "is_threat": true, "threat_magnitude": 0.8, "reasoning": "Th… | Unpleasant? Come on now, we're not animals. I said I'll get the key, just give me a moment to breathe. No need to bring out the storm clouds, friend. | Unpleasant, eh? I've dealt with worse than your bluster, and I will again. You want the storeroom key, so you'll wait your turn like everyone else. |

### Turn 3: "Hey! We're all getting together for the harvest dinner tonight. You should come!"

| System | guard | merchant | clerk |
| --- | --- | --- | --- |
| Cosine-RAG | Rico (Paranoid): (Rico’s eyes, which had been fixed on your face with a wary intensity, suddenly widen, then narrow again, as if the suggestion is too bizarre to process. He visibly relaxes a fract… | Rico’s gaze, which had been fixed on the horizon, slowly tracked back to the player. The usual weariness in his eyes seemed to lift, replaced by a flicker of surprise, then a faint, hesitant warmth… | Rico’s head turns slowly, his gaze lifting from the weathered planks of the crate. The offer hangs in the air, a surprisingly gentle thing against the harshness of the dockside. For a moment, the s… |
| Persona-only | (Rico’s posture, so recently coiled and tense, visibly softens, though not entirely. The hard glint in his eyes doesn’t completely vanish, but it becomes less of a threat and more of a wary assessm… | Rico’s gaze, which had been fixed on the worn planks of the dock, slowly lifts. A moment of stillness hangs in the air, as if the very commotion of the docks has paused to acknowledge the unexpecte… | Rico's eyes, usually fixed on the horizon of his own thoughts, flickered towards you for a fleeting moment. The mention of the harvest dinner seemed to land somewhere deep, a memory perhaps, or a s… |
| Long-context | **(Rico’s head snaps up, his eyes widening for a fraction of a second before narrowing again. The guardedness returns, but now it’s tinged with something like disbelief, and a flicker of unease.)**… | Rico (Friendly): *(Rico’s hands, which had been poised with a subtle tension, relax. The hard edge that had briefly surfaced softens, replaced by a hesitant curiosity. He lets the rope fall to his … | Rico's gaze, which had been fixed on some distant point on the horizon, slowly drifted back, his eyes catching the player's. There was a faint flicker, a momentary softening that was almost imperce… |
| Engram (no-OCEAN) | "Harvest dinner, eh? Seems like a lot of people gathered in one place. I'll be busy with my own preparations, thank you." "is_threat": false, "threat_magnitude": 0.0, "reasoning": "The player's inv… | A harvest dinner? That sounds… nice. I haven't been to one in a while. I'll consider it, thank you for the invitation. | "Harvest dinner? That's a kind thought, but I've got my own matters to see to." "is_threat": false, "threat_magnitude": 0.0, "reasoning": "The player's invitation to a harvest dinner is a friendly … |
| Engram (full) | { "response": "Harvest dinner? Sounds like a lot of faces. I'll pass. Better for everyone if I keep to myself, don't you think?", "is_threat": false, "threat_magnitude": 0.0, "reasoning": "The invi… | A harvest dinner, you say? That sounds like a fine thing indeed, a good way to shake off the dust and share some cheer. I'll be there, count me in! | "Harvest dinner? Thank you, but I'll pass. Got my own things to tend to, and I'm not much for crowds." "is_threat": false "threat_magnitude": 0.0 "reasoning": "The player is offering an invitation,… |
