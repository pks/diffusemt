# Discrete Diffusion Notes

## Why Switch from Continuous
Continuous diffusion had two unsolved problems:
1. Can't generate from pure Gaussian noise (5% accuracy at t=199)
2. Nearest-neighbor embedding→token projection is lossy and non-differentiable

## Three Approaches Designed

### A. Mask-Only (standard MDLM-style)
- Concat [source | target], source never masked, target masked with [MASK]
- Encoder-only bidirectional transformer
- Pro: well-studied. Con: starts from empty target.

### B. Source-as-Corruption
- Single sequence = target tokens, replace with source tokens as "noise"
- Pro: semantically meaningful start. Con: requires same length src/tgt.

### C. Hybrid (IMPLEMENTED — chosen)
- Concat [source | target], source always visible, target masked with [MASK]
- Encoder-only, segment embeddings (0=src, 1=tgt)
- Different lengths supported, confidence-based unmasking at inference
- Pro: flexible, source always available. Con: 2x sequence length.

## Implementation Details (Approach C)
- model.py: DiffusionTransformer — single TransformerEncoder, tied output weights
- diffusion.py: MaskDiffusion — gamma(t) = 1 - cos(pi/2 * t/T)^2
- Loss: CE on masked target positions only, in fp32 (fp16 overflows with large vocab)
- Logits scaled by 1/sqrt(embed_dim) for tied embeddings
- AdamW optimizer, cosine LR decay after linear warmup

## Collapse Still Happens (v13-v17)
All discrete runs collapse. Pattern: t=1 accuracy suddenly drops to 0% (1 unique token)
while t=200 accuracy is still improving. Happens when LR reaches a threshold.

| Run | lr | Extras | Collapsed at |
|-----|-----|--------|-------------|
| v13 (1st) | 2e-4 | - | step 2000 |
| v13 (2nd) | 1e-4 | - | step 12500 (best run) |
| v14 | 1e-4 | cosine decay | step 5000 |
| v15 | 1e-4 | untied weights | step 1000 |
| v16 | 3e-5 | label_smooth=0.1 | step 5000 |
| v17 | 3e-5 | 29K vocab | step 500 |

## Hypotheses for Collapse
1. **Tied embedding feedback loop**: CE gradient → embeddings → input destabilized
2. **Larger models collapse faster** (untied 148M collapsed at step 1000 vs tied 87M at 12500)
3. **Smaller vocab doesn't help** — 29K collapsed even faster than 120K

## Untried Ideas
1. Freeze token_embedding, only train transformer + untied output_proj
2. Initialize from pretrained BERT, freeze embeddings
3. Bottleneck output: project d→256→vocab instead of d→vocab
4. Fine-tune pretrained seq2seq model with masking objective
5. Try approach B (source-as-corruption)
