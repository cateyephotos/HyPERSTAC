#!/usr/bin/env bash
# Train VICReg encoders one after another inside WSL, detached from the Claude app.
#   setsid nohup bash run_train_queue.sh "runs/tonsil31_paper8:0" "runs/tonsil31_all32:0" > runs/train_queue.log 2>&1 &
# Each job is "<run>:<fold>". A job that crashes is retried and resumes from its last epoch
# checkpoint; the queue stops when two consecutive attempts end with the same error line.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/wsl_env.sh"
cd "$HERE"
BATCH="${BATCH:-48}"; EPOCHS="${EPOCHS:-100}"; CACHE="${CACHE:-$HOME/hyperstac_cache}"
for job in "$@"; do
  run="${job%%:*}"; fold="${job##*:}"
  out="$run/ssl/fold-$fold"; mkdir -p "$out"
  if [ -f "$out/DONE" ]; then echo "$(date -Iseconds) skip $job (done)"; continue; fi
  last=""
  for attempt in $(seq 1 "${MAX_ATTEMPTS:-6}"); do
    echo "$(date -Iseconds) start $job attempt $attempt"
    python -m hyperstac_tool.train_vicreg --run "$run" --fold "$fold" --epochs "$EPOCHS" \
      --batch-size "$BATCH" --tile-cache "$CACHE" >> "$out/train_stdout.log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$out/DONE" ]; then echo "$(date -Iseconds) done $job"; break; fi
    # Last real error line. Exclude TensorFlow's oneDNN banner, which mentions
    # "round-off errors" and would make every crash look identical.
    err="$(grep -E 'Error|OOM|Segmentation fault|Aborted|double free' "$out/train_stdout.log" \
           | grep -v 'oneDNN' | tail -1)"
    echo "$(date -Iseconds) $job failed rc=$rc: $err"
    if [ -n "$last" ] && [ "$err" = "$last" ]; then echo "STOP: identical error twice"; exit 2; fi
    last="$err"
  done
done
echo "$(date -Iseconds) queue finished"
