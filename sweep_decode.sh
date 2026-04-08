#!/bin/bash
CKPT="checkpoints_v32/model_final.pt"
PY="/workspace/.venv/bin/python3"
N=500

echo "=== Baseline: temp=2.0, 200 steps ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 2.0

echo "=== temp=1.0 ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 1.0

echo "=== temp=1.5 ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 1.5

echo "=== temp=3.0 ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 3.0

echo "=== temp=2.0 + stochastic + anneal ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 2.0 --stochastic --anneal-temperature

echo "=== temp=2.0 + rerank 5 ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 8 --temperature 2.0 --rerank 5

echo "=== temp=2.0 + 50 steps ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 2.0 --num-steps 50

echo "=== temp=2.0 + refine 50 ==="
$PY eval_bleu.py --checkpoint $CKPT --n $N --batch-size 32 --temperature 2.0 --refine-steps 50

echo "SWEEP COMPLETE"
