#!/bin/bash
# Wait for step 100K, then report val loss trend and run health check

echo "Monitoring v25 training for step 100K..."

while true; do
    if grep -q "Step 100000 | Val Loss" /workspace/v25_train.log 2>/dev/null; then
        echo "=== Step 100K reached ==="
        grep "Val Loss" /workspace/v25_train.log | tail -15
        
        # Check if checkpoint exists
        if ls /workspace/checkpoints_v25_mask_diffusion/checkpoint_100000.pt 2>/dev/null; then
            echo "Checkpoint 100K exists"
        fi
        
        # Get latest health check
        grep -A5 "Health check" /workspace/v25_train.log | tail -10
        break
    fi
    sleep 60
done
