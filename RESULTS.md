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
| **v33** | **Encoder-only 32L×512d** | **Unfrozen mBERT embeddings** | **20.71** | temp=1.0, 200 steps | 195.3M trainable; init from v32; new best |

## Key Findings

1. **Depth >> Width**: 512d×32L (v32: 19.62) > 512d×24L (v26: 18.41) >> 768d×24L (v31: 15.83)
2. **~18.4 BLEU ceiling at 24L**: Weighted loss, self-conditioning, knowledge distillation all converged to ~18.4
3. **Depth broke the ceiling**: v32's 32 layers pushed from 18.4 → 19.6 (+1.2 BLEU)
4. **Unfrozen embeddings broke it again**: v33 unfreezing mBERT embeddings (differential LR 1e-5) pushed 19.6 → 20.7 (+1.1 BLEU)
5. **Decoding tuning is marginal**: Sweep showed max +0.1 BLEU from decoding params alone
6. **Reference**: MarianMT (autoregressive) = 23.59 BLEU; SOTA discrete diffusion = ~25-27 BLEU

## v32 Decoding Sweep (500 sentences)

| Config | BLEU | Delta |
|--------|------|-------|
| temp=2.0 (default) | 19.66 | baseline |
| **temp=1.0** | **19.77** | **+0.11** |
| temp=1.5 | 19.63 | -0.03 |
| temp=3.0 | 19.73 | +0.07 |
| 50 steps (vs 200) | 19.69 | +0.03 |
| refine 50 | 18.83 | -0.83 |
