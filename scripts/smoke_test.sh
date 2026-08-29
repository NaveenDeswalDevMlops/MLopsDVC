#!/usr/bin/env bash
# End-to-end smoke test against a running API: liveness, readiness, model identity,
# a real prediction with a real image, and the metrics endpoint.
# Usage: smoke_test.sh [base_url]
set -euo pipefail

BASE="${1:-http://127.0.0.1:8000}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0
FAIL=0

check() {
  local name="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf '  ok   %s\n' "$name"; PASS=$((PASS + 1))
  else
    printf '  FAIL %s\n' "$name"; FAIL=$((FAIL + 1))
  fi
}

echo "smoke test against $BASE"

check "GET /health returns 200" curl -fsS "$BASE/health"
check "GET /ready returns 200"  curl -fsS "$BASE/ready"
check "GET /model-info returns 200" curl -fsS "$BASE/model-info"
check "GET /metrics returns 200" curl -fsS "$BASE/metrics"

IMAGE="$(find "$ROOT/data/processed/test" -name '*.jpg' 2>/dev/null | head -1 || true)"
if [ -n "$IMAGE" ]; then
  BODY="$(curl -fsS -X POST "$BASE/predict" -F "file=@${IMAGE};type=image/jpeg" || echo '')"
  if echo "$BODY" | grep -q '"label"'; then
    printf '  ok   POST /predict returned a label: %s\n' "$(echo "$BODY" | head -c 120)"
    PASS=$((PASS + 1))
  else
    printf '  FAIL POST /predict did not return a label\n'; FAIL=$((FAIL + 1))
  fi
else
  printf '  skip POST /predict (no processed test images; run make preprocess)\n'
fi

# A malformed body must be rejected rather than silently classified.
STATUS="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H 'Content-Type: application/json' -d '{"image_base64":"not-base64"}')"
if [ "$STATUS" = "422" ]; then
  printf '  ok   POST /predict rejects a malformed payload with 422\n'; PASS=$((PASS + 1))
else
  printf '  FAIL malformed payload returned %s, expected 422\n' "$STATUS"; FAIL=$((FAIL + 1))
fi

echo ""
echo "passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
