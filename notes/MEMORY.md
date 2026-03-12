# DiffuseMT Project Memory

## Project Overview
Diffusion-based machine translation (English->German) using WMT14 data.
Codebase: ~1000 lines Python. Key files: config.py, model.py, diffusion.py, train.py, translate.py, dataset.py, eval.py

## Hardware (original)
- 2x NVIDIA TITAN RTX (24GB each)
- Training: DDP with `torchrun --nproc_per_node=2 train.py`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` required for large models
- User is moving to a beefier GPU machine

## Current Status: Discrete Diffusion (v13-v17), All Collapsed

### Phase 1: Continuous Diffusion (v1-v12b) — ABANDONED
- Only 512d/8L with T=200 (v8) avoided collapse, but can't generate from noise (5% at t=199)
- T=50 causes universal collapse; larger models collapse even with T=200

### Phase 2: Discrete Diffusion (v13+) — CURRENT
Switched to absorbing-state mask-based diffusion. Architecture: encoder-only transformer,
[source | masked_target] concatenated, segment embeddings, cross-entropy loss.
See [discrete_diffusion.md](discrete_diffusion.md) for approach design.

**All discrete runs collapse.** Best run (v13, lr=1e-4) survived 12.5K steps with
t=200 acc reaching 14% before collapsing. Collapse always hits t=1 first (easy task).

Tried: lower LR, cosine decay, label smoothing, untied weights, smaller vocab (29K).
None prevented collapse. Smaller vocab and untied weights made it worse.

**Current hypothesis**: Tied embedding feedback loop — CE loss gradient updates embeddings
used for both input and output, destabilizing input representations over time.

### Possible Next Steps
1. Freeze token_embedding (only train transformer + output_proj)
2. Use pretrained BERT embeddings (frozen)
3. Bottleneck output head (project to small dim before vocab)
4. Fine-tune a pretrained model instead of training from scratch
5. Revisit approach B (source-as-corruption)

## Code State (on `discrete` branch)
- model.py: encoder-only transformer, tied weights, 1/sqrt(d) logit scaling
- diffusion.py: MaskDiffusion — absorbing state, confidence-based unmasking
- config.py: currently set to v17 (bert-base-cased, lr=3e-5, label_smoothing=0.1)
- dataset.py: points to bert-base-cased tokenized data
- train.py: CE loss on masked positions, cosine LR decay, AdamW

## Key Gotchas
- Tokenizer: currently bert-base-cased (29K vocab), was bert-base-multilingual-cased (120K)
- Python env: .venv/bin/python (.venv in project root)
- User wants pure diffusion model (no CE auxiliary loss)
- DDP port conflicts: wait a few seconds between training runs
- **Logit scaling**: tied embeddings need `/ sqrt(embed_dim)` or logits blow up (std≈22)
- Health check collapse detection at t=1 (barely masked) is the key diagnostic

## Detailed References
- [training_history.md](training_history.md) — all versions v1-v17, both phases
- [discrete_diffusion.md](discrete_diffusion.md) — discrete diffusion design & approaches
- [gpu_profiling.md](gpu_profiling.md) — GPU memory profiling data (TITAN RTX)
