#!/usr/bin/env bash
# Poll a URL until it answers 200, or give up.
# Usage: wait_for.sh <url> [attempts]
set -euo pipefail
URL="${1:?usage: wait_for.sh <url> [attempts]}"
ATTEMPTS="${2:-30}"
for attempt in $(seq 1 "$ATTEMPTS"); do
  if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
    echo "up: $URL (after ${attempt}s)"
    exit 0
  fi
  sleep 1
done
echo "timed out waiting for $URL after ${ATTEMPTS}s" >&2
exit 1
