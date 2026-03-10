# DiffuseMT Project Memory

## Project Overview
Diffusion-based machine translation (English->German) using WMT14 data.
Codebase: ~1000 lines Python. Key files: config.py, model.py, diffusion.py, train.py, translate.py, dataset.py, eval.py

## Hardware (original)
- 2x NVIDIA TITAN RTX (24GB each)
- Training: DDP with `torchrun --nproc_per_node=2 train.py`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required for large models
- User is moving to a beefier GPU machine

## Two Critical Unsolved Problems

### 1. Mode Collapse — Caused by T=50, Not Model Size
**CRITICAL**: The baseline_test proved that 512d/8L with T=50 ALSO collapses.
The only non-collapsing run (v8) used T=200. ALL T=50 runs collapsed regardless of size.
T=200 is necessary but not sufficient — large models (v6/v7) collapsed with T=200 too,
but those had other bugs. Next step: re-run larger models with T=200 + all current fixes.
See [training_history.md](training_history.md) for full collapse matrix.

### 2. Can't Generate From Noise
Even the working 512d/8L model can't translate — it's a good denoiser (100% token
accuracy at t<120) but can't generate from pure noise (5% at t=199). Reverse diffusion
starts from pure noise, so output is garbage. Infilling also fails.
Trained up to 150K steps (v9), t>180 accuracy didn't improve.

## Training Recipe (proven for 512d/8L/T=200 only)
- **T=200 timesteps minimum** — T=50 causes universal collapse
- x0 prediction (model predicts clean embeddings, not noise)
- emb_scale = weight.std() (~1.0)
- Cosine noise schedule: alpha_bar = cos(pi/2 * t/T)^2
- Min-SNR loss weighting: gamma=5.0
- LR warmup: 2000 optimizer steps linear warmup
- Per-sample MSE / embed_dim normalization
- Per-layer gradient checkpointing, AMP fp16 + GradScaler
- Output LayerNorm before projection (added v12, kept)

## Code Features
- `train.py --resume <ckpt>`: checkpoint resumption with optimizer state
- Auto collapse detection: health checks at steps 500/1K/2K/5K + every val_every
- `eval.py --checkpoint <ckpt>`: per-timestep accuracy, translation, infilling eval
- Structured metrics in `<checkpoint_dir>/metrics.jsonl`

## Detailed References
- [training_history.md](training_history.md) — all versions v1-v12b, collapse analysis
- [gpu_profiling.md](gpu_profiling.md) — GPU memory profiling data (TITAN RTX)

## Key Gotchas
- Tokenizer: bert-base-multilingual-cased (~120K vocab)
- Python env: .venv/bin/python (.venv in project root)
- User wants pure diffusion model (no CE auxiliary loss)
- DDP port conflicts: wait a few seconds between training runs
- Health check cdist OOMs on GPU for large models — runs token matching on CPU
