# DiffuseMT Project Memory

## Project Overview
Diffusion-based machine translation (English->German) using WMT14 data.
Codebase: ~1000 lines Python. Key files: config.py, model.py, diffusion.py, train.py, translate.py, dataset.py, eval.py

## Hardware
- 2x NVIDIA TITAN RTX (24GB each)
- Training: DDP with `torchrun --nproc_per_node=2 train.py`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required for large models
- Python env: use `uv run python` (not `.venv/bin/python` directly)

## Current Status: v18 Pretrained BERT — FIRST WORKING MODEL

### The Solution: Pretrained BERT + Frozen Embeddings
- `bert-base-multilingual-cased` (12L/768d/12H, 209.4M params, 117.6M trainable)
- Frozen word embeddings break the tied-embedding feedback loop
- Untied bottleneck output head: LayerNorm → Linear(768,256) → GELU → Linear(256,119547)
- Sinusoidal timestep conditioning added to hidden states
- Gradient checkpointing via `encoder.gradient_checkpointing = True`

### Training (v18)
- DDP 2 GPUs, batch_size=64/GPU, grad_accum=4, effective batch=512
- lr=5e-5, warmup=2000 steps, cosine decay, T=200 timesteps
- Training complete at 100K steps: val_loss=3.26, t=200 acc=22.7%, unique=482
- Produces real German translations and infilling
- Final checkpoint: `checkpoints_v18_pretrained/model_final.pt`
- Intermediate checkpoints at steps 10K, 20K, 30K, 40K, 50K, 60K, 70K, 90K

### Sampling Fix
- `p_sample_loop` now estimates target length from source (×1.5)
- Initializes as `[CLS] [MASK]... [SEP] [PAD]...` (not all-MASK)
- Fixes CLS/SEP/PAD positions, only unmasks middle positions

### Previous Phases (all collapsed — see training_history.md)
- Phase 1: Continuous diffusion (v1-v12b) — abandoned, can't generate from noise
- Phase 2: Discrete diffusion (v13-v17) — all collapsed due to tied embedding feedback loop

## Key Gotchas
- Tokenizer: bert-base-multilingual-cased (120K vocab)
- Collapse detection uses t=T/2 (not t=1, which has near-zero masking with cosine schedule)
- DDP port conflicts: wait a few seconds between training runs
- User wants pure diffusion model (no CE auxiliary loss)

## Detailed References
- [training_history.md](training_history.md) — all versions v1-v18
- [discrete_diffusion.md](discrete_diffusion.md) — discrete diffusion design & approaches
- [gpu_profiling.md](gpu_profiling.md) — GPU memory profiling data (TITAN RTX)
