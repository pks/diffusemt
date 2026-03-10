# Training History

## Version Summary

| Ver | Dir | Config | Params | Steps | Final Loss | Result |
|-----|-----|--------|--------|-------|------------|--------|
| v1 | checkpoints_v1/ | 768d/12L | 294M | 100K | ~1.0 | Loss stuck (epsilon pred + bad emb_scale) |
| v2 | checkpoints_v2/ | experimental | - | 20K | - | Abandoned |
| v3 | checkpoints_v3/ | 768d/12L | 294M | 30K | - | Abandoned |
| v4 | checkpoints/ | 1024d/28L | 951M | 50K | ~1.0 | Loss stuck (epsilon pred + bad emb_scale) |
| v5 | checkpoints_v5/ | 1024d/28L | 951M | 50K | ~1.0 | emb_scale fixed, but still epsilon pred |
| v6 | checkpoints_v6/ | 1024d/28L | 951M | 50K | 0.994 | x0 pred, mode collapse ("ький") |
| v7 | checkpoints_v7/ | 1024d/28L | 951M | ~40K | ~0.21 | Cosine+min-SNR, mode collapse ([CLS]) |
| v7b | checkpoints_v7b/ | 1024d/28L | 951M | ~25K | - | Mode collapse ("寔") |
| v8 | checkpoints_v8/ | 512d/8L | ~115M | 50K | 0.214 | No collapse, good denoiser, bad generator |
| v9 | checkpoints_v9/ | 512d/8L | ~115M | ~150K | 0.178 | Continued v8, still garbage translations |
| v10 | checkpoints_v10/ | 1024d/28L/T50 | 951M | ~45K | 0.82 | Plateaued immediately, mode collapse |
| v10b | checkpoints_v10b/ | 1024d/28L/T50 | 951M | ~150 | 1.06 | lr=1e-4/warmup10K, too slow (5 step/min) |
| v11 | checkpoints_v11/ | 768d/12L/T50 | 294M | 500 | 0.95 | Collapse at step 500 (health check OOM bug) |
| v11b | checkpoints_v11b/ | 768d/12L/T50 | 294M | 1000 | 0.92 | lr=3e-5/warmup10K, collapse at step 1000 |
| v11c | checkpoints_v11c/ | 768d/12L/T50 | 294M | 500 | 0.85 | lr=2e-4/warmup2K (same as v8), collapse |
| v12 | checkpoints_v12/ | 768d/12L/T50 | 294M | 500 | 0.93 | +LayerNorm+anchor loss, still collapsed |
| v12b | checkpoints_v12b/ | 512d/24L/T50 | ~340M | 500 | 0.85 | Deeper 512d, still collapsed |

## Bug Fixes Applied (cumulative)

1. **emb_scale** (v5): `weight.norm(dim=-1).mean()` -> `weight.std()` (~32 -> ~1.0)
2. **x0 prediction** (v6): changed loss from epsilon to x0 target
3. **Cosine schedule** (v7): `alpha_bar = cos(pi/2 * t/T)^2`
4. **Min-SNR weighting** (v7): `w(t) = min(SNR(t), gamma) / SNR(t)`, gamma=5.0
5. **/embed_dim normalization** (v7): divide per-sample MSE by embed_dim
6. **LR warmup** (v8): linear warmup over 2000 optimizer steps
7. **Output LayerNorm** (v12): nn.LayerNorm before output_proj

## Infrastructure Added
- **train.py --resume**: checkpoint resumption with optimizer state
- **Health checks**: auto collapse detection at steps 500/1K/2K/5K + every val_every
- **eval.py**: per-timestep accuracy, translation, infilling, quantitative infilling accuracy
- **metrics.jsonl**: structured logging in checkpoint dir

## Critical Finding: Mode Collapse Scales With Model Size

**Only 512d/8L (~115M params) avoids mode collapse.** Every larger configuration collapses:

| Config | Params | lr | Warmup | Collapsed? |
|--------|--------|----|--------|------------|
| 512d/8L | 115M | 2e-4 | 2K | NO |
| 512d/24L | 340M | 2e-4 | 2K | YES (step 500) |
| 768d/12L | 294M | 2e-4 | 2K | YES (step 500) |
| 768d/12L | 294M | 1e-4 | 10K | YES (step 500) |
| 768d/12L | 294M | 3e-5 | 10K | YES (step 1000) |
| 768d/12L+LN+anchor | 294M | 2e-4 | 2K | YES (step 500) |
| 1024d/28L | 951M | 2e-4 | 2K | YES (plateaued) |
| 1024d/28L | 951M | 1e-4 | 10K | YES (5 step/min, too slow) |

This is NOT a hyperparameter issue — collapse happens at every lr and warmup tested.
LayerNorm and anchor loss also didn't help.

The collapse pattern: model predicts a single token for all positions at all timesteps,
even for clean input (t=0). Unique token count drops to 1-5.

**Possible root causes to investigate:**
- Embedding dimension vs vocab size ratio: 512d/120K vocab = 0.004 dims per token,
  768d/120K = 0.006. Larger models may have more capacity to overfit the mean.
- The MSE loss landscape may have a stronger collapse attractor in higher dimensions.
- Self-conditioning might amplify collapse: if first prediction collapses, self-cond
  reinforces it.
- Gradient checkpointing interaction with larger models.

## v8/v9 Inference Analysis

v8 (50K) and v9 (150K) both produce garbage translations despite low training loss.

**Per-timestep accuracy (v9 @ 150K):**
| Timestep | Token Accuracy |
|----------|---------------|
| 0-120 | 100% |
| 150 | 94.8% |
| 160 | 91.9% |
| 170 | 79.7% |
| 180 | 51.6% |
| 190 | 14.6% |
| 199 | 5.8% |

Model is excellent denoiser but can't generate from pure noise.
More training (50K→150K) improved mid-range (t=150-170) but t>180 stayed flat.

**Infilling also fails**: even with prefix+suffix context, the 5-token gaps are
filled with garbage. The reverse diffusion process itself is broken for generation.

## Recommendations for Next Machine (Beefier GPUs)

1. **Investigate collapse root cause** before scaling up — it's not just lr/warmup
2. **Try disabling self-conditioning** for larger models (might amplify collapse)
3. **Try different output head**: e.g., predict logits then lookup embeddings instead
   of directly predicting embedding vectors
4. **Consider freezing token_embedding** during training (it gets optimizer updates
   even though x0 target is detached)
5. **Try progressive growing**: train 512d/8L, then expand to wider model
6. **Reduce timesteps** (T=50 instead of 200) was already done, didn't help collapse
7. If generation from noise stays broken, consider **non-autoregressive MT**
   approaches like CMLM that don't require starting from pure noise
