# Discrete Diffusion Translation — BLEU Results (newstest2014 EN→DE)

All scores on the **full 3003-sentence** WMT14 newstest2014 test set unless noted.

## Best BLEU by Version

| Version | Architecture | Key Change | BLEU | Decoding | Notes |
|---------|-------------|------------|------|----------|-------|
| v13–v17 | various | — | — | — | All collapsed, no BLEU evals |
| v18 | mBERT encoder-only (full pretrained) | First working model | — | — | No full eval (qualitative only) |
| v20 | mBERT encoder-only | Mask diffusion | — | — | 200-sentence only: 7.59 |
| v21 | From-scratch enc-dec, frozen mBERT embed | Architecture switch | — | — | 200-sentence only: 9.40 |
| v22 | From-scratch enc-dec, frozen mBERT embed | Deeper enc-dec | — | — | 200-sentence only: 13.30 |
| v24 | Encoder-only 12L×512d | Diffusion-only (no AR) | — | — | 200-sentence only: 12.73 |
| v25 | Encoder-only 12L×512d | Mask diffusion (MDLM-style) | — | — | 200-sentence only: 13.60 |
| **v26** | **Encoder-only 24L×512d** | **Depth scaling (from v25)** | **18.41** | temp=1.5, 50 steps | First full eval; 77.9M trainable |
| v27b | Encoder-only 24L×512d | Weighted loss + length pred | 18.36 / 13.11 | temp=2.0 | 18.36 w/o length pred; 13.11 with (harmful) |
| v28 | Encoder-only 24L×512d | Self-conditioning | ~18.4 | — | No improvement over v26 |
| v29 | Encoder-only 24L×512d | Knowledge distillation | 18.39 | temp=2.0 | MarianMT teacher; no gain |
| v30b | Encoder-only 24L×768d | Width scaling | — | — | Crashed at 100K; 200-sent only: 13.99 |
| v31 | Encoder-only 24L×768d | Width scaling + fixes | 15.83 | — | Wider = worse (-2.6 BLEU vs v26) |
| **v32** | **Encoder-only 32L×512d** | **More depth (from v26)** | **19.62** | temp=2.0, 200 steps | 103.5M trainable; best result |
| v32 | | | 19.42 | temp=1.0, 200 steps | Lower temp slightly worse on full set |
| v33 | Encoder-only 32L×512d | Unfrozen mBERT embeddings | 20.71 | temp=1.0, 200 steps | 195.3M trainable; init from v32; 200K steps |
| **v33 ext** | **Encoder-only 32L×512d** | **Extended training (300K)** | **20.95** | temp=1.0, 200 steps | Best at 285K (val_loss 1.3659); new best |
| **v34** | **Encoder-only 40L×512d** | **Depth scaling to 40L** | **21.18** | temp=1.0, 200 steps | 129.2M trainable; init from v33; step 165K best |
| v34 | | | 21.02 | temp=1.0, 200 steps | Step 200K (final); slight overfit |
| v35 | Encoder-only 32L×512d | CFG (10% source dropout) | ~21.2* | temp=1.0, no CFG | *500-sent only; CFG guidance HURTS at all weights |
| v36 | Encoder-only 42L×768d | Native 768d (no projection) | 14.45* | temp=1.0, 200 steps | *500-sent; 301M params; from scratch; val_loss 1.71 but BLEU regression |

## Key Findings

1. **Depth >> Width**: 512d×32L (v32: 19.62) > 512d×24L (v26: 18.41) >> 768d×24L (v31: 15.83)
2. **~18.4 BLEU ceiling at 24L**: Weighted loss, self-conditioning, knowledge distillation all converged to ~18.4
3. **Depth broke the ceiling**: v32's 32 layers pushed from 18.4 → 19.6 (+1.2 BLEU)
4. **Unfrozen embeddings broke it again**: v33 unfreezing mBERT embeddings (differential LR 1e-5) pushed 19.6 → 20.7 (+1.1 BLEU)
5. **Extended training helps marginally**: v33 ext (200K→300K) gained +0.24 BLEU (20.71→20.95); best checkpoint at 285K, not 300K
6. **40 layers beats 32**: v34's 40L achieved 21.18 despite worse val_loss (1.41 vs 1.37) — val_loss ≠ BLEU
7. **CFG hurts**: v35 trained with 10% source dropout for classifier-free guidance. At inference, any CFG weight > 0 degrades BLEU monotonically (21.2 → 16.2 at w=3.0)
8. **Native 768d hurts BLEU**: v36 removed 768→512 projection, ran at mBERT's native 768d×42L. Val_loss 1.71 (better than v26's ~1.80) but BLEU 14.45 — massive regression. Training from scratch at 768d for 200K steps is insufficient; cascaded init (v26→v32→v33→v34) provides 500K+ effective steps
9. **Decoding tuning is marginal**: Sweep showed max +0.1 BLEU from decoding params alone
10. **Reference**: MarianMT (autoregressive) = 23.59 BLEU; SOTA discrete diffusion = ~25-27 BLEU

## v32 Decoding Sweep (500 sentences)

| Config | BLEU | Delta |
|--------|------|-------|
| temp=2.0 (default) | 19.66 | baseline |
| **temp=1.0** | **19.77** | **+0.11** |
| temp=1.5 | 19.63 | -0.03 |
| temp=3.0 | 19.73 | +0.07 |
| 50 steps (vs 200) | 19.69 | +0.03 |
| refine 50 | 18.83 | -0.83 |

## v35 CFG Weight Sweep (500 sentences)

| CFG Weight | BLEU (step 70K) | BLEU (final) | Delta |
|------------|-----------------|--------------|-------|
| **0 (baseline)** | **21.17** | **21.19** | — |
| 0.5 | 20.43 | 20.45 | -0.74 |
| 1.0 | 19.16 | 19.57 | -1.6 to -2.0 |
| 1.5 | 18.47 | 18.53 | -2.7 |
| 2.0 | 17.64 | 17.77 | -3.4 |
| 3.0 | 16.17 | 16.42 | -4.8 to -5.0 |

CFG monotonically degrades BLEU. The unconditional P(target) mode pushes translations away from source-conditioned content.

---

## Final Summary

**Best model**: v34 — encoder-only 40L×512d, mask diffusion, unfrozen mBERT embeddings, **21.18 BLEU** on WMT14 EN→DE newstest2014 (3003 sentences). This is 89.8% of the MarianMT autoregressive baseline (23.59 BLEU).

### What worked

| Technique | BLEU gain | Versions |
|-----------|-----------|----------|
| Pretrained mBERT (solved collapse) | 0 → working | v18 |
| Mask diffusion over source-as-corruption | +0.9 | v24→v25 |
| Depth scaling (12→24→32→40 layers) | +5.0 → +1.2 → +0.2 | v25→v26→v32→v34 |
| Cascaded initialization (layer stacking) | enables depth scaling | v26, v32, v34 |
| Unfreezing mBERT embeddings (differential LR) | +1.1 | v32→v33 |
| Extended training / LR restarts | +0.2 | v33→v33 ext |

### What didn't work

| Technique | Result | Versions |
|-----------|--------|----------|
| Continuous diffusion (Gaussian noise) | Universal collapse or can't generate | v1–v12b |
| From-scratch discrete diffusion | Universal collapse | v13–v17 |
| Width scaling (512d→768d) | -2.6 BLEU | v30b, v31 |
| Self-conditioning | No BLEU gain despite better health metrics | v28 |
| Knowledge distillation (MarianMT teacher) | No gain | v29 |
| Weighted loss on masked positions | Better val_loss, same BLEU | v27b |
| Length predictor | Actively harmful (-5.3 BLEU) | v27b |
| Classifier-free guidance | Monotonically degrades BLEU | v35 |
| Native 768d (no projection) | -6.7 BLEU (from-scratch bottleneck) | v36 |

### Architecture evolution

```
v1-v12b: continuous diffusion (abandoned — universal collapse at T=50)
    ↓
v13-v17: discrete diffusion from scratch (abandoned — collapse)
    ↓
v18: pretrained mBERT encoder-only (solved collapse, first translations)
    ↓
v19-v20: source-as-corruption (works but inferior to mask diffusion)
    ↓
v21-v23: from-scratch encoder-decoder + AR warmup (BLEU 13.30)
    ↓
v24: encoder-only, diffusion-only with curriculum (12.73)
    ↓
v25: mask diffusion (13.60) → v26: 24L (18.41) → v32: 32L (19.62)
    ↓
v33: unfrozen embeddings (20.71) → v33 ext: 300K steps (20.95)
    ↓
v34: 40L (21.18) ← BEST
    ↓
v35: CFG (negative) / v36: 768d (negative) ← diminishing returns
```

### Key takeaways

1. **Pretrained embeddings are essential.** From-scratch discrete diffusion collapses; frozen mBERT embeddings break the tied-embedding feedback loop.
2. **Depth >> width.** Every depth increase improved BLEU; every width increase hurt it. The 768→512 projection is a beneficial bottleneck, not a limitation.
3. **Cascaded initialization is critical.** The v34 best model has 500K+ effective training steps through the v25→v26→v32→v33→v34 init chain. Training the same architecture from scratch (v36) massively underperforms.
4. **Val_loss ≠ BLEU.** v34 has worse val_loss than v33 ext (1.41 vs 1.37) but better BLEU (21.18 vs 20.95). Deeper models generate more coherent sequences even with higher per-token loss.
5. **The gap to AR is structural.** At ~21 BLEU vs ~24 AR, the remaining gap likely requires fundamental changes (better diffusion objectives, iterative refinement, or hybrid AR-diffusion) rather than more scaling of the current approach.
