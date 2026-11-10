#!/usr/bin/env bash
# Source inside WSL before running GPU stages:  source /mnt/e/Downloads_6-13-25/HyPERSTAC/wsl_env.sh
# TensorFlow 2.21 does not find the pip-installed NVIDIA libraries on its own (it reports
# "Cannot dlopen some GPU libraries", e.g. libcusolver.so.11), so put them on the path.
VENV="$HOME/hyperstac_tf"
SP="$VENV/lib/python3.12/site-packages/nvidia"
LIBS=""
for d in "$SP"/*/lib; do LIBS="$LIBS:$d"; done
export LD_LIBRARY_PATH="${LIBS#:}:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TF_CPP_MIN_LOG_LEVEL=2
export MPLBACKEND=Agg
export PATH="$VENV/bin:$PATH"
