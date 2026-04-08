#!/bin/bash
# CFG weight sweep for v35
CKPT="${1:-checkpoints_v35/model_final.pt}"
PY="/workspace/.venv/bin/python3"
N=500
GPU="${2:-0}"

echo "=== Checkpoint: $CKPT ==="
echo "=== GPU: $GPU, N=$N ==="

echo "=== Baseline: temp=1.0, 200 steps, no CFG ==="
CUDA_VISIBLE_DEVICES=$GPU $PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 1.0

echo "=== CFG w=0.5 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 16 --temperature 1.0 --cfg-weight 0.5

echo "=== CFG w=1.0 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 16 --temperature 1.0 --cfg-weight 1.0

echo "=== CFG w=1.5 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 16 --temperature 1.0 --cfg-weight 1.5

echo "=== CFG w=2.0 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 16 --temperature 1.0 --cfg-weight 2.0

echo "=== CFG w=3.0 ==="
CUDA_VISIBLE_DEVICES=$GPU $PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 16 --temperature 1.0 --cfg-weight 3.0

echo "SWEEP COMPLETE"
