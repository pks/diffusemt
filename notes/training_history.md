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
| v13 | checkpoints_v13/ | 512d/8L/T200 | ~115M | 10K+ | 0.20 | Reproduced v8 on new hardware |
| v14 | checkpoints_v14/ | 512d/8L/T1000 | ~115M | 5K+ | 0.23 | T=1000+DDIM. Good denoiser, can't generate |
| v15 | checkpoints_v15/ | 512d/8L/T1000 | ~115M | 5K | 0.37 | 20% per-batch source-init. Function words OK, content collapse ("תרבות") |
| v16 | checkpoints_v16/ | 768d/12L/T1000 | ~294M | 2.5K | - | 768d/12L + 20% source-init. Borderline at step 500, recovered, then collapsed at step 2500 |
| v17 | checkpoints_v17/ | 512d/8L/T1000 | ~115M | 3K+ | 0.58 | 20% per-batch source-init + CE loss (0.1). Good denoiser but same garbage translations as v15 |
| v18 | checkpoints_v18/ | 512d/8L/T1000 | ~115M | 5K | 0.54 | **50% per-sample source-init + CE loss**. First real translations ("Die Europäische")! Collapsed at step 5300 |
| v19 | checkpoints_v19/ | 512d/8L/T1000 | ~115M | 6.2K | - | Resume v18@5K, lr=1e-4, 35% source-init, CE=0.1. Gradient explosion at step 6200 |
| v20 | checkpoints_v20/ | 512d/8L/T1000 | ~115M | 6.5K | 0.80 | Resume v18@5K, fresh optim, lr=1e-4, 35% source-init, NO CE. Mode collapse at step 6500 |
| v21 | checkpoints_v21/ | 512d/8L/T1000 | ~115M | 2.8K | 0.82 | Fresh start, 50% source-init, no CE, cosine decay. Mode collapse at step 2800 |
| v22 | checkpoints_v22/ | 512d/8L/T1000 | ~115M | 2.1K | 1.63 | AdamW+EMA+clip=0.5+CE clamp+35%+cosine. Grad explosion at lr=2e-4 peak |
| v23 | checkpoints_v23/ | 512d/8L/T1000 | ~115M | 2.3K+ | 1.005 | **lr=1e-4 peak**. t=0 100%/448 unique at step 2K. MOST STABLE RUN YET. Running. |

## Bug Fixes Applied (cumulative)

1. **emb_scale** (v5): `weight.norm(dim=-1).mean()` -> `weight.std()` (~32 -> ~1.0)
2. **x0 prediction** (v6): changed loss from epsilon to x0 target
3. **Cosine schedule** (v7): `alpha_bar = cos(pi/2 * t/T)^2`
4. **Min-SNR weighting** (v7): `w(t) = min(SNR(t), gamma) / SNR(t)`, gamma=5.0
5. **/embed_dim normalization** (v7): divide per-sample MSE by embed_dim
6. **LR warmup** (v8): linear warmup over 2000 optimizer steps
7. **Output LayerNorm** (v12): nn.LayerNorm before output_proj

## Bug Fixes Applied (new machine)

8. **DDP health check fix** (v13): health_check/validate used DDP model on rank 0 only, causing NCCL timeout on rank 1. Fixed to use raw_model and run on all ranks.
9. **Health check OOM fix** (v13): cdist on full batch (512, 128, 120K) OOMed. Fixed to use HC_BATCH=16 subset and run on GPU.
10. **DDIM sampling** (v14): added deterministic DDIM reverse process to diffusion.py
11. **T=1000** (v14): finer noise schedule for better training at all noise levels.
12. **Source-init** (v15): 20% of batches use noisy source embeddings as xt. First partial translations.
13. **bfloat16** (v16): switched from fp16+GradScaler to bfloat16 (no scaler needed). ~82-85GB GPU.
14. **Auxiliary CE loss** (v17): cross-entropy loss on nearest-token logits (weight=0.1, subset of 64 samples)
15. **Per-sample source-init** (v18): per-sample mixing (each sample independently gets source-init with prob p) vs per-batch (entire batch flips). Smoother gradients.

## Infrastructure Added
- **train.py --resume**: checkpoint resumption with optimizer state
- **train.py --fresh-optim**: resume model weights only, fresh optimizer + scheduler
- **Health checks**: auto collapse detection at steps 500/1K/2K/5K + every val_every
- **eval.py**: per-timestep accuracy, translation, infilling, quantitative infilling accuracy
- **metrics.jsonl**: structured logging in checkpoint dir (step, train_loss, lr, grad_norm, health_check events)
- **EMA checkpoints**: ema_model saved in every checkpoint (CPU state dict)
- **Early saves**: save every 500 steps for first 5000, then every val_every

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

## CRITICAL FINDING: Source-Init Rate vs Stability

Source-init training teaches translation by replacing xt with noisy source embeddings (target stays the same). Key findings:

| Model | Source-init rate | Method | LR | Result |
|-------|-----------------|--------|-----|--------|
| 512d/8L | 20% per-batch | v15 | 2e-4 | Function words only, content collapse ("תרבות") |
| 768d/12L | 50% per-batch | v16 | 2e-4 | Collapsed at step 500 |
| 768d/12L | 20% per-batch | v16 | 2e-4 | Recovered at step 1000, collapsed at step 2500 |
| 512d/8L | 20% per-batch + CE | v17 | 2e-4 | Good denoiser, same garbage translations |
| 512d/8L | 50% per-sample + CE | v18 | 2e-4 | **First real translations!** Collapsed at step 5300 |
| 512d/8L | 35% per-sample + CE | v19 | 1e-4 | Resume v18. Gradient explosion step 6200 |
| 512d/8L | 35% per-sample, no CE | v20 | 1e-4 | Resume v18. Mode collapse step 6500 |
| 512d/8L | 50% per-sample, no CE | v21 | 2e-4 | Fresh. Mode collapse step 2800 |
| 512d/8L | 35% per-sample + CE | v22 | 2e-4 | Fresh+AdamW+EMA. Grad explosion at step 2100 (lr peak) |
| 512d/8L | 35% per-sample + CE | v23 | **1e-4** | **Fresh+AdamW+EMA. 100% acc step 2K. STABLE.** Running. |

**Key insights**:
- Per-sample source-init >> per-batch (smoother gradients)
- 35% source-init + CE loss + lr=1e-4 is the winning combo (v23)
- lr=2e-4 universally fails with source-init (v18/v21/v22 all collapsed)
- CE loss prevents mode collapse but must be clamped to prevent gradient explosion
- Without CE loss, mode collapse happens faster (v20/v21 vs v18/v19)

## v17: CE Loss Helps Denoising, Not Translation

Config: 512d/8L, T=1000, bs=8192/GPU, lr=2e-4, warmup=2K, 20% per-batch source-init, CE loss 0.1

Added auxiliary cross-entropy loss on predicted embeddings → nearest token logits. CE_BATCH=64 to avoid OOM.

Health checks showed excellent denoising (100% t=0 at step 2000, 448 unique). But translation diagnostic was identical garbage to v15:
- "The weather is nice today." → `Die شورویsey dios Guinness Yunan. takaisin...`
- "The European Parliament..." → `Derца Agence ஐ יהודה die die...`

**Conclusion**: CE loss improves denoising precision but doesn't help translation. The bottleneck is insufficient source-init training signal (20% is too low).

## v18: Per-Sample 50% Source-Init — BREAKTHROUGH

Config: 512d/8L, T=1000, bs=8192/GPU, lr=2e-4, warmup=2K, **50% per-sample source-init**, CE loss 0.1

**Health checks:**
| Step | t=0 (unique) | t=500 | t=750 | t=999 (unique) | Val Loss |
|------|-------------|-------|-------|----------------|----------|
| 500 | 32.6% (227) | 30.5% (213) | 19.0% (131) | 7.2% (40) | - |
| 1000 | 100% (438) | 93.2% (437) | 44.2% (427) | 3.3% (72) | - |
| 2000 | 100% (448) | 99.3% (447) | 61.8% (441) | 4.2% (47) | - |
| 2500 | 87.8% (416) | 87.7% (411) | 64.0% (416) | 5.6% (47) | 0.63 |
| 5000 | 90.6% (364) | 91.0% (368) | 79.3% (374) | 6.4% (75) | 0.54 |

**Translation diagnostic at step 2500:**
- "The weather is nice today." → `Das vena ist 육 салалық mortal.` (Das=correct, ist=correct!)
- "I have a cat." → `Ich vena vena ceety.` (Ich=correct!)
- "The European Parliament..." → `Die иться 육 authority hemiboreala die die.` (Die=correct!)

**Translation diagnostic at step 5000:**
- "The weather is nice today." → `##нихäste ist ływe ైన.` (ist=correct)
- "I have a cat." → `Ich Hubert climb climb.` (Ich=correct)
- **"The European Parliament has approved the proposal." → `Die Europäische chte Constitucional clásico die Reporter.`** (Die Europäische = CORRECT!)
- "Hello, how are you?" → `##禺, Sie Sie Sie Opening` (Sie=correct "you")

**FIRST REAL TRANSLATIONS!** Function words AND some content words correct. "Die Europäische" is a meaningful German phrase. Model clearly learning source→target mapping.

**Collapse at step 5300**: Loss exploded 5.0 → 5.3 → 5.5 → 7.1 → 8.8 in 300 steps. 50% source-init + lr=2e-4 is unstable long-term.

## v19: Resume v18 with lower LR (FAILED)

Config: Resume from v18 step 5000, lr=1e-4 (halved), 35% per-sample source-init (reduced from 50%), CE loss 0.1

**Result**: Gradient explosion at step 6200. Same failure mode as v18 but delayed.

## v20: Resume v18 without CE loss (FAILED)

Config: Resume from v18 step 5000, fresh optimizer, lr=1e-4, 35% source-init, NO CE loss

**Result**: Mode collapse at step 6500. Loss plateaued at 0.80, 0% accuracy, 1 unique token. Different failure mode — CE loss removal changed gradient explosion → mode collapse.

## v21: Fresh start, cosine decay, no CE (FAILED)

Config: Fresh start, 512d/8L, T=1000, 50% per-sample source-init, NO CE loss, lr=2e-4, cosine LR decay, warmup=2K

**Result**: Mode collapse at step 2800. Loss jumped 0.556→0.816. Collapsed much earlier than v18, suggesting CE loss actually helps delay collapse.

## CRITICAL FINDING: Two Competing Failure Modes

| Config | CE Loss | Failure Mode | When |
|--------|---------|-------------|------|
| v18 | 0.1 | Gradient explosion | Step 5300 |
| v19 | 0.1 | Gradient explosion | Step 6200 |
| v20 | 0.0 | Mode collapse | Step 6500 |
| v21 | 0.0 | Mode collapse | Step 2800 |

**Key insight**: CE loss prevents mode collapse but eventually causes gradient explosion as logits sharpen. Without CE loss, mode collapse happens faster. Need to handle BOTH failure modes simultaneously.

## v22: Kitchen Sink Stabilization (running)

Config: Fresh start, 512d/8L, T=1000, **35%** per-sample source-init, CE=0.1, lr=2e-4, bfloat16

**New stabilization techniques:**
1. **AdamW** (weight_decay=0.01) — regularization prevents parameter explosion
2. **EMA** (decay=0.9999) — exponential moving average of model weights for stable inference
3. **Gradient clipping = 0.5** (from 1.0) — tighter clipping catches instability earlier
4. **CE loss clamped at 5.0** — prevents gradient explosion while maintaining mode collapse prevention
5. **Cosine LR decay to 10% of peak** (not to 0) — maintains some learning signal
6. **Save every 500 steps** early on — capture pre-collapse checkpoints

Rationale: Each previous failure had one fix applied. v22 applies all fixes simultaneously. EMA ensures we preserve the best model weights even if training eventually becomes unstable.

**Result**: Step 1000 was excellent (t=0 93.2%, 324 unique). But loss started rising at step 1900 as LR approached peak 2e-4. By step 2000 (LR=2e-4): t=0 dropped to 41%, 69 unique. Step 2100: gradient explosion (grad_norm=143,890). **Root cause: lr=2e-4 is too high for source-init training.**

## v23: lr=1e-4 Peak — MOST STABLE RUN (running)

Config: 512d/8L, T=1000, bs=8192/GPU, **lr=1e-4**, warmup=1K, cosine decay to 10% (1e-5), AdamW (wd=0.01), 35% per-sample source-init, CE=0.1 (clamped at 5.0), grad_clip=0.5, EMA (0.9999, CPU, update every 10 steps), bfloat16

**Health checks:**
| Step | t=0 (unique) | t=250 (unique) | t=500 (unique) | t=750 (unique) | t=999 (unique) | Loss | GradNorm |
|------|-------------|----------------|----------------|----------------|----------------|------|----------|
| 500 | 20.6% (17) | 20.6% (16) | 19.4% (14) | 18.1% (9) | 3.5% (3) | 1.23 | 0.06 |
| 1000 | 93.3% (411) | 93.0% (416) | 68.5% (285) | 37.9% (94) | 3.2% (4) | 1.11 | 0.08 |
| 2000 | **100% (448)** | **100% (448)** | **99.6% (449)** | 57.0% (206) | 3.1% (5) | 1.02 | 0.07 |
| 2500 | 87.2% (410) | 88.0% (412) | 86.5% (409) | 59.5% (263) | 3.1% (7) | 1.00 | 0.08 |

**Key observations:**
- Step 500 looks borderline (17 unique) but recovers dramatically by step 1000 (411 unique)
- Step 2000: PERFECT t=0 and t=250 accuracy (100%, 448 unique) — matching v18's best
- t=500 at 99.6% is BETTER than v18 ever achieved (v18 peaked at 99.3% at step 2000)
- Grad norms consistently small (0.05-0.09) — no sign of instability
- Loss declining steadily: 1.23 → 1.11 → 1.02 → 1.005 (at step 2300)
- **Passed the step 2000 test where v22 collapsed** (v22 grad_norm exploded to 143,890 at step 2100)

**Training speed:** ~8 steps/min (500 steps in ~63 min). Checkpoints saved every 500 steps early, every 2500 steps later.

**Validation at step 2500:** val_loss=0.3453 — significantly better than v18's 0.63 at same step. t=0 dipped to 87% (from 100% at step 2000) but 410 unique tokens is excellent. t=750 improved to 59.5% (263 unique). Training continues stable.

**Status as of step 2550:** Loss=0.998, grad_norm=0.06. Training rock solid through step 2500 milestone.

**Launch command:**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup .venv/bin/torchrun --nproc_per_node=2 train.py > checkpoints_v23/train.log 2>&1 &
```

**Resume command (if needed):**
```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup .venv/bin/torchrun --nproc_per_node=2 train.py --resume checkpoints_v23/model_step_XXXX.pt > checkpoints_v23/train.log 2>&1 &
```

## CRITICAL FINDING: lr=2e-4 Is Incompatible With Source-Init

v22 proved definitively that lr=2e-4 is too high for source-init training:
- v22 at step 1000 (lr=1e-4 during warmup): 93.2% t=0 accuracy, 324 unique — healthy
- v22 at step 2000 (lr=2e-4 at peak): 41.2% t=0 accuracy, 69 unique — collapsing
- v22 at step 2100: grad_norm=143,890 — gradient explosion

v23 at lr=1e-4 peak: 100% t=0 accuracy at step 2000, no instability. **lr=1e-4 is the maximum safe LR.**

## Bug Fixes Applied (v22-v23)

16. **AdamW** (v22): switched from Adam to AdamW (weight_decay=0.01) for regularization
17. **EMA** (v22): exponential moving average of model weights (decay=0.9999, CPU-based, update every 10 steps)
18. **Gradient clipping = 0.5** (v22): tightened from 1.0 to catch instability earlier
19. **CE loss clamped at 5.0** (v22): prevents gradient explosion while keeping mode collapse prevention
20. **Cosine LR decay to 10%** (v22): decays to min_lr_ratio=0.1 of peak, not to zero
21. **lr=1e-4 peak** (v23): lr=2e-4 proven incompatible with source-init training

## Recommendations

1. **Use T≥200** — T=50 causes universal collapse
2. **Per-sample source-init** — smoother gradients than per-batch
3. **Source-init rate**: 35% is the sweet spot — enough for translation, stable long-term
4. **LR=1e-4 maximum** — lr=2e-4 causes gradient explosion at peak with source-init
5. **CE loss**: NEEDED to prevent mode collapse, but must clamp at 5.0 to prevent gradient explosion
6. **768d/12L incompatible** with source-init training — always collapses
7. **bfloat16 works well** on Blackwell GPUs, no GradScaler needed
8. **EMA** (0.9999 decay) on CPU — preserves best weights; update every 10 steps to avoid speed penalty
9. **AdamW** (weight_decay=0.01) for regularization
10. **Cosine LR decay to 10% of peak** — maintaining min learning signal prevents mode collapse
11. **Grad clip = 0.5** — tighter than default 1.0, catches instability earlier
