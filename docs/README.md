# Engram — Memory Replay (Visualization)

Static GitHub Pages site that replays Engram session event logs. Pure HTML + CSS + vanilla JS, no build step.

## What it shows

For each NPC session, the page replays in order:

1. **OCEAN radar** — baseline (dashed) vs. effective (filled), animating on `profile_decay` and pulsing red on `fight_flight_applied`.
2. **Pipeline timeline** — one card per turn, with the six pipeline stages (threat / retrieval / mode / contradiction / response / consolidate) filling in as events fire. Click any card to expand the full retrieval table sorted by personality-weighted score, with the threshold line drawn in.
3. **Memory store** — rolling 7-slot session window, the full memory list (new memories flash green; retrieved IDs glow yellow), and key memories + Prolog facts (asserted=green, revised=yellow with strikethrough, rejected=red).
4. **Transcript** — player + NPC dialogue, scroll-locked to playback.

Group sessions (same `group` field in the manifest) render side-by-side under a single shared playback timeline.

## Deploy

1. Push this repo to GitHub.
2. Repo → Settings → Pages.
3. Source: **Deploy from a branch**, branch: `main`, folder: `/docs`.
4. Save. The URL will be something like `https://<user>.github.io/<repo>/`.

The site also works directly opened from `file://` and via `python -m http.server -d docs`.

## Where the logs come from

Engram's CLI writes NDJSON event logs when run with `--viz` (or with `ENGRAM_VIZ_LOG=1` set). Each session goes to `docs/sessions/<npc_id>-<timestamp>.ndjson` and gets registered in `docs/sessions/manifest.json`.

The CLI/log writer lives in the parallel Python workstream — this site only consumes the schema.

## Manifest format

`docs/sessions/manifest.json`:

```json
{
  "sessions": [
    {
      "id": "demo-jeanie-2026-04-29",
      "npc_id": "jeanie",
      "npc_name": "Jeanie",
      "file": "sample-session.ndjson",
      "group": null
    }
  ]
}
```

Field meanings:

- `id` — unique slug, used as the dropdown value.
- `npc_id`, `npc_name` — display labels in the selector and panel header.
- `file` — relative path under `docs/sessions/`.
- `group` — string. Sessions sharing a group are a multi-NPC run (same player input fed to multiple presets); the UI renders them side-by-side. `null` is a standalone session.

## Multi-NPC group convention

When two or more sessions share the same `group` value, they were produced by Engram's multi-NPC run mode (one player input distributed across several preset NPCs). Selecting any of them in the dropdown switches to a side-by-side layout with one column per NPC, all driven by a single shared playback clock.

## Files

```
docs/
├── index.html
├── style.css
├── app.js                 — control flow / playback engine
├── js/
│   ├── state.js           — pure event-to-state reducer
│   └── render.js          — idempotent DOM renderer
├── sessions/
│   ├── manifest.json
│   └── sample-session.ndjson
└── README.md
```

## Event schema

See `ARCHITECTURE.md` in the repo root for the pipeline; the event types consumed here are:

`session_init`, `session_init_npc`, `turn_start`, `embedding_done`, `threat_assessed`, `retrieval_scored`, `mode_selected`, `fight_flight_applied`, `contradiction_check`, `response_generated`, `consolidated`, `memory_added`, `summary_added`, `key_promoted`, `fact_asserted`, `fact_revised`, `fact_rejected`, `profile_decay`, `turn_end`, `session_end`, `session_end_npc`.

Each line of the NDJSON is a single JSON object with `t` (seconds since session start), `type`, and `payload`.
