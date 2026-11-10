#!/usr/bin/env bash
# After the training queue exits, embed and cluster every finished encoder.
#   setsid nohup bash run_post.sh runs/tonsil31_paper8 runs/tonsil31_all32 > runs/post.log 2>&1 &
# Waits while run_train_queue.sh is running (the GPU is busy), then processes each
# <run>/ssl/fold-<k> that has DONE but no spc/spc_qc.json. Safe to rerun.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/wsl_env.sh"
cd "$HERE"
while pgrep -f "bash run_train_queue.sh" > /dev/null; do sleep 60; done
echo "$(date -Iseconds) training queue not running; starting post-processing"
for run in "$@"; do
  for d in "$run"/ssl/fold-[0-9]; do
    [ -f "$d/DONE" ] || { echo "skip $d (no encoder)"; continue; }
    fold="${d##*fold-}"
    if [ ! -f "$d/embeddings.npz" ]; then
      echo "$(date -Iseconds) embed $d"
      python -m hyperstac_tool.embed --run "$run" --fold "$fold" >> "$d/post_stdout.log" 2>&1 || { echo "embed failed $d"; continue; }
    fi
    if [ ! -f "$d/spc/spc_qc.json" ]; then
      echo "$(date -Iseconds) cluster $d"
      python -m hyperstac_tool.cluster --run "$run" --fold "$fold" >> "$d/post_stdout.log" 2>&1 || echo "cluster failed $d"
    fi
    [ -f "$d/spc/spc_qc.json" ] && echo "$(date -Iseconds) ok $d: $(cat "$d/spc/spc_qc.json" | tr -d '\n ' | cut -c1-200)"
  done
done
echo "$(date -Iseconds) post-processing finished"
