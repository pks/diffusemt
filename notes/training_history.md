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
| baseline_test | checkpoints_baseline_test/ | 512d/8L/T50 | ~115M | 500 | - | **COLLAPSED** — same as v8 but T=50 instead of T=200 |

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

## CRITICAL FINDING: T=50 Causes Mode Collapse (NOT Model Size)

**CORRECTION**: The earlier hypothesis that "only 512d/8L avoids collapse" was WRONG.
The baseline_test run proved that 512d/8L with T=50 ALSO collapses at step 500 (4 unique tokens).

The ONLY run that avoided collapse was v8: 512d/8L with **T=200**. ALL T=50 runs collapsed:

| Config | Params | T | lr | Warmup | Collapsed? |
|--------|--------|---|-----|--------|------------|
| 512d/8L | 115M | **200** | 2e-4 | 2K | **NO** (only survivor) |
| 512d/8L | 115M | **50** | 2e-4 | 2K | **YES** (step 500) — baseline_test |
| 512d/24L | 340M | 50 | 2e-4 | 2K | YES (step 500) |
| 768d/12L | 294M | 50 | 2e-4 | 2K | YES (step 500) |
| 768d/12L | 294M | 50 | 1e-4 | 10K | YES (step 500) |
| 768d/12L | 294M | 50 | 3e-5 | 10K | YES (step 1000) |
| 768d/12L+LN+anchor | 294M | 50 | 2e-4 | 2K | YES (step 500) |
| 1024d/28L | 951M | 50 | 2e-4 | 2K | YES (plateaued) |
| 1024d/28L | 951M | 50 | 1e-4 | 10K | YES (5 step/min, too slow) |

Note: v6/v7/v7b (T=200, large models) also collapsed, so T=200 is necessary but not sufficient.
The interaction is: T=50 collapses everything; T=200 gives larger models a chance but they
still collapse (possibly due to separate model-size issues).

The collapse pattern: model predicts a single token for all positions at all timesteps,
even for clean input (t=0). Unique token count drops to 1-5.

**Root cause analysis:**
- T=50 with cosine schedule means huge noise jumps between steps — the model can't
  learn the gradual denoising trajectory.
- With T=200, smaller models (512d/8L) can learn, but larger models still collapse —
  likely due to the MSE loss landscape having a stronger collapse attractor in higher dimensions.
- Self-conditioning may amplify collapse: if first prediction collapses, self-cond reinforces it.
- Gradient checkpointing interaction with larger models is also possible.

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

1. **Use T=200 (or higher), NOT T=50** — T=50 causes universal collapse. This is the #1 lesson.
2. **Try scaling up with T=200**: v6/v7 collapsed at 1024d/28L/T=200, but those runs had
   other bugs (v6 had emb_scale issues, v7 was early). Re-run 768d or 1024d with T=200
   and all current fixes (cosine schedule, min-SNR, LayerNorm, health checks).
3. **Try disabling self-conditioning** for larger models (might amplify collapse)
4. **Try different output head**: e.g., predict logits then lookup embeddings instead
   of directly predicting embedding vectors
5. **Consider freezing token_embedding** during training (it gets optimizer updates
   even though x0 target is detached)
6. **Try T=500 or T=1000** — if T=200 helps vs T=50, more timesteps might help further
7. If generation from noise stays broken, consider **non-autoregressive MT**
   approaches like CMLM that don't require starting from pure noise
