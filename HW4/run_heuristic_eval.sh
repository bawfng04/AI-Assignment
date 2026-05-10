#!/bin/bash
# ./run_heuristic_eval.sh > heuristic.log 2>&1
set -e

if [ ! -d ".venv" ]; then
    if command -v python3.11 &> /dev/null; then
        python3.11 -m venv .venv
    elif command -v python3.10 &> /dev/null; then
        python3.10 -m venv .venv
    else
        python3 -m venv .venv
    fi
fi

source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install "gymnasium[atari]" ale-py

mkdir -p checkpoints runs
export CUDA_VISIBLE_DEVICES=0

python train.py --env ALE/Pong-v5 --device cuda