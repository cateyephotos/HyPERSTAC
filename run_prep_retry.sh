#!/usr/bin/env bash
# Retry wrapper for hyperstac_tool prep. Each attempt resumes from verified shards.
# Stops early when two consecutive attempts fail with the identical last error line
# (workstation rule: retry transient interpreter faults, not deterministic bugs).
set -u
PY="$(dirname "$0")/.venv_tool/Scripts/python.exe"
OUT=""; for a in "$@"; do [ "${prev:-}" = "--out" ] && OUT="$a"; prev="$a"; done
mkdir -p "$OUT"
last=""
for attempt in $(seq 1 ${MAX_ATTEMPTS:-8}); do
  echo "=== attempt $attempt $(date -Iseconds)" >> "$OUT/attempts.log"
  "$PY" -m hyperstac_tool prep "$@" > "$OUT/attempt_$attempt.log" 2>&1
  rc=$?
  tail -2 "$OUT/attempt_$attempt.log" >> "$OUT/attempts.log"
  [ $rc -eq 0 ] && { echo "OK after $attempt attempt(s)" >> "$OUT/attempts.log"; exit 0; }
  err="$(grep -E 'Error|error' "$OUT/attempt_$attempt.log" | tail -1)"
  if [ -n "$last" ] && [ "$err" = "$last" ]; then
    echo "STOP: identical error twice: $err" >> "$OUT/attempts.log"; exit 2
  fi
  last="$err"
done
exit 1
