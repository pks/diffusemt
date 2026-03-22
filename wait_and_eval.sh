#!/bin/bash
# Wait for step 5000 checkpoint, then run eval
cd /home/shadeform/exp/diffusemt
CKPT="checkpoints_v15/model_step_5000.pt"

while [ ! -f "$CKPT" ]; do
    LAST_STEP=$(tail -1 checkpoints_v15/metrics.jsonl 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['step'])" 2>/dev/null || echo "?")
    echo "$(date '+%H:%M:%S') - waiting for $CKPT (last step: $LAST_STEP)"
    sleep 120
done

echo "=== Checkpoint found! Running eval ==="
CUDA_VISIBLE_DEVICES=1 .venv/bin/python eval.py --checkpoint "$CKPT" --device cuda 2>&1
