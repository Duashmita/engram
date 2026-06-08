#!/usr/bin/env bash
# Usage: backend/smoke.sh [BACKEND_URL]
# Default URL: http://localhost:8000
#
# End-to-end smoke test of the live Engram demo backend's API contract:
#   1. GET  /health         -> 200, {"ok": true}
#   2. POST /start          -> 200, captures session_id
#   3. POST /turn (SSE)     -> stream until "type":"turn_end"
#   4. POST /end            -> 204
#   5. POST /turn (stale)   -> 404 (session was cleaned up)

set -eEuo pipefail

BACKEND_URL="${1:-${BACKEND_URL:-http://localhost:8000}}"
STEP=0
SESSION_ID=""

on_error() {
    local rc=$?
    echo "FAIL: step ${STEP}" >&2
    exit "${rc}"
}
trap on_error ERR

have_jq() { command -v jq >/dev/null 2>&1; }

extract_field() {
    # extract_field <json> <field>
    local json="$1" field="$2"
    if have_jq; then
        printf '%s' "${json}" | jq -r --arg f "${field}" '.[$f] // empty'
    else
        printf '%s' "${json}" \
            | grep -o "\"${field}\"[[:space:]]*:[[:space:]]*\"[^\"]*\"" \
            | head -n1 \
            | awk -F'"' '{print $4}'
    fi
}

pretty() {
    if have_jq; then
        printf '%s' "$1" | jq .
    else
        printf '%s\n' "$1"
    fi
}

# ---------------------------------------------------------------------------
# Step 1 — health
# ---------------------------------------------------------------------------
STEP=1
echo "[1/5] GET ${BACKEND_URL}/health"
HEALTH_BODY="$(curl -fsS "${BACKEND_URL}/health")"
pretty "${HEALTH_BODY}"
if have_jq; then
    OK="$(printf '%s' "${HEALTH_BODY}" | jq -r '.ok // false')"
else
    OK="$(printf '%s' "${HEALTH_BODY}" | grep -o '"ok"[[:space:]]*:[[:space:]]*true' || true)"
fi
case "${OK}" in
    true|*'"ok"'*) ;;
    *) echo "health did not report ok=true" >&2; false ;;
esac

# ---------------------------------------------------------------------------
# Step 2 — start
# ---------------------------------------------------------------------------
STEP=2
echo "[2/5] POST ${BACKEND_URL}/start"
START_BODY="$(curl -fsS -X POST "${BACKEND_URL}/start" \
    -H 'Content-Type: application/json' \
    -d '{"npc_id":"jeanie"}')"
pretty "${START_BODY}"
SESSION_ID="$(extract_field "${START_BODY}" "session_id")"
if [ -z "${SESSION_ID}" ]; then
    echo "no session_id in /start response" >&2
    false
fi
echo "session_id=${SESSION_ID}"

# ---------------------------------------------------------------------------
# Step 3 — turn (SSE stream)
# ---------------------------------------------------------------------------
STEP=3
echo "[3/5] POST ${BACKEND_URL}/turn (SSE stream)"
TURN_PAYLOAD="$(printf '{"session_id":"%s","player_input":"hey, how'\''s it going?"}' "${SESSION_ID}")"
# Stream and stop at the first turn_end event. `|| true` keeps set -e happy
# when curl is killed by SIGPIPE on the awk-side close.
{
    curl -fsS -N -X POST "${BACKEND_URL}/turn" \
        -H 'Content-Type: application/json' \
        -H 'Accept: text/event-stream' \
        -d "${TURN_PAYLOAD}" || true
} | awk '
    /^data: / { print; fflush() }
    /"type"[[:space:]]*:[[:space:]]*"turn_end"/ { exit 0 }
'

# ---------------------------------------------------------------------------
# Step 4 — end (expect 204)
# ---------------------------------------------------------------------------
STEP=4
echo "[4/5] POST ${BACKEND_URL}/end"
END_CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BACKEND_URL}/end" \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"${SESSION_ID}\"}")"
echo "http=${END_CODE}"
[ "${END_CODE}" = "204" ] || { echo "expected 204, got ${END_CODE}" >&2; false; }

# ---------------------------------------------------------------------------
# Step 5 — stale /turn (expect 404)
# ---------------------------------------------------------------------------
STEP=5
echo "[5/5] POST ${BACKEND_URL}/turn with stale session (expect 404)"
STALE_CODE="$(curl -s -o /dev/null -w '%{http_code}' -X POST "${BACKEND_URL}/turn" \
    -H 'Content-Type: application/json' \
    -d "{\"session_id\":\"${SESSION_ID}\",\"player_input\":\"still there?\"}")"
echo "http=${STALE_CODE}"
[ "${STALE_CODE}" = "404" ] || { echo "expected 404, got ${STALE_CODE}" >&2; false; }

echo "OK"
