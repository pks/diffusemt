# Training History

## Phase 1: Continuous Diffusion (v1-v12b, baseline_test)

Continuous Gaussian diffusion over token embeddings. Model predicts clean embeddings (x0),
nearest-neighbor lookup converts back to tokens. Encoder-decoder architecture.

### Version Summary (Continuous)

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

### Continuous Diffusion Findings

**T=50 causes universal collapse.** The only surviving run (v8) used T=200.
v8/v9 trained well but can't generate from pure noise (5% accuracy at t=199).
Continuous diffusion abandoned due to these two unsolvable problems.

---

## Phase 2: Discrete Diffusion (v13+)

**Forked from continuous at this point.** Switched to absorbing-state (mask-based) discrete
diffusion. Motivation: avoids the embedding→token projection gap and starts from [MASK]
tokens instead of pure Gaussian noise.

### Architecture: Approach C (Hybrid)
- **Encoder-only** bidirectional transformer (no encoder-decoder split)
- Input: `[source tokens | corrupted target tokens]` concatenated
- Source tokens are **never masked** — always visible as context
- Target tokens are masked by diffusion process (replace with [MASK])
- Segment embeddings (0=source, 1=target) distinguish the two halves
- Source and target can have **different lengths** (both padded to max_seq_len=128)
- Output: softmax over vocab, cross-entropy loss on masked target positions
- Confidence-based unmasking during inference (unmask most-confident first)

### Three Approaches Considered (see discrete_diffusion.md)
- **A. Mask-only**: standard MDLM-style, concat [src|tgt], mask target
- **B. Source-as-corruption**: replace target tokens with source tokens (same length required)
- **C. Hybrid** (chosen): concat [src|tgt], source always visible, target masked. Different lengths OK.

### Version Summary (Discrete)

| Ver | Dir | Config | Tokenizer | lr | Extras | Steps | Result |
|-----|-----|--------|-----------|-----|--------|-------|--------|
| v13 (1st) | checkpoints_v13_discrete/ | 512d/8L, tied, T=200 | mbert-120K | 2e-4 | - | 2000 | Collapsed at step 2000 (LR hit peak) |
| v13 (2nd) | checkpoints_v13_discrete/ | 512d/8L, tied, T=200 | mbert-120K | 1e-4 | - | 12500 | **Best run.** Collapsed at step 12500 when LR hit full value. t=200 acc reached 14%, val_loss 7.85 |
| v14 | checkpoints_v14_cosine_decay/ | 512d/8L, tied, T=200 | mbert-120K | 1e-4 | cosine decay | 5000 | Collapsed step 5000 (still in warmup, decay hadn't kicked in) |
| v15 | checkpoints_v15_untied/ | 512d/8L, **untied**, T=200 | mbert-120K | 1e-4 | cosine decay | 1000 | Collapsed step 1000 — untied is worse (148M params, bigger = faster collapse) |
| v16 | checkpoints_v16_smooth/ | 512d/8L, tied, T=200 | mbert-120K | 3e-5 | cosine decay + label_smooth=0.1 | 5000 | Collapsed step 5000 — lower LR + smoothing didn't help |
| v17 | checkpoints_v17_smallvocab/ | 512d/8L, tied, T=200 | bert-cased-29K | 3e-5 | cosine decay + label_smooth=0.1 | 500 | Collapsed step 500 — smaller vocab made it worse |

### Key Bug Fix: Logit Scaling for Tied Embeddings
With tied output weights (`output_proj.weight = token_embedding.weight`), logits have
std ≈ sqrt(embed_dim) ≈ 22.6 for d=512. This blows up the softmax and gives initial CE
loss of ~98 (vs expected ~11.7). Fix: divide logits by `sqrt(embed_dim)`.

### v13 (2nd run) Detailed Trajectory — Best Discrete Run
| Step | Loss | t=1 acc | t=200 acc | t=200 unique | Val Loss |
|------|------|---------|-----------|--------------|----------|
| 500 | 11.09 | 100% | 4.9% | 3 | - |
| 1000 | 10.34 | 100% | 7.3% | 4 | - |
| 2000 | 10.05 | 100% | 8.8% | 7 | - |
| 2500 | 9.92 | 100% | 9.3% | 8 | 9.84 |
| 5000 | 9.34 | 100% | 11.8% | 11 | 9.34 |
| 7500 | 8.63 | 100% | 12.7% | 25 | 8.63 |
| 10000 | 7.85 | 100% | 13.3% | 34 | 7.85 |
| 12500 | - | **0% (1 unique)** | 13.8% | 55 | - | COLLAPSED |

Collapse always happens at t=1 (almost no masking) — the model suddenly predicts a
single token for all positions even when input is barely corrupted. t=200 accuracy
was actually improving at the time of collapse.

### Discrete Collapse Analysis

**Pattern**: Every discrete run eventually collapses. The collapse correlates with
LR magnitude — higher LR = earlier collapse. But even lr=3e-5 collapsed.

Attempted fixes that didn't work:
- Lower LR (2e-4 → 1e-4 → 3e-5): delayed but didn't prevent collapse
- Cosine LR decay: collapsed before decay kicked in
- Untied output weights: made it worse (more params = faster collapse)
- Label smoothing (0.1): no effect
- Smaller vocab (120K → 29K): made it worse

**Hypothesis**: The tied embedding creates a feedback loop. The CE loss gradient
updates the embedding table, which also provides the input representations. As the
model improves, it pushes embeddings apart to create sharper predictions, which
destabilizes the input distribution. The 1/sqrt(d) logit scaling helps but doesn't
fully prevent this.

### v18: Pretrained BERT — SOLVED COLLAPSE

**Architecture**: Pretrained `bert-base-multilingual-cased` (12L/768d/12H) with:
- Frozen word embeddings (91.8M params frozen)
- Unfrozen encoder layers, position/segment/LayerNorm embeddings
- Untied bottleneck output head: LayerNorm → Linear(768,256) → GELU → Linear(256,119547)
- Sinusoidal timestep conditioning added to hidden states
- Gradient checkpointing (BERT's built-in `encoder.gradient_checkpointing = True`)
- 209.4M total params, 117.6M trainable

**Training config**:
- DDP on 2x TITAN RTX, batch_size=64/GPU, grad_accum=4 → effective batch 512
- lr=5e-5, warmup=2000 steps, cosine decay to 100K steps
- Mixed precision (fp16), AdamW, grad clip 1.0
- T=200 timesteps, cosine schedule

**Key fix**: Collapse detection changed from t=1 (unreliable with cosine schedule near-zero masking)
to t=T/2 (50% masking). False positive collapse at step 15000 was caused by
t=1 having only 1 masked position in the entire batch (larger batch=64 vs 48).

| Step | Train Loss | Val Loss | t=200 acc | t=200 unique |
|------|-----------|----------|-----------|--------------|
| 2,500 | 7.51 | 7.53 | 12.45% | 4 |
| 5,000 | 6.42 | 6.51 | 13.24% | 27 |
| 10,000 | 5.15 | 5.28 | 16.70% | 147 |
| 15,000 | 4.49 | 4.63 | 17.32% | 253 |
| 20,000 | 4.18 | 4.27 | 17.40% | 301 |
| 25,000 | 4.04 | 3.94 | 17.77% | 340 |
| 30,000 | 3.90 | 3.87 | 17.44% | 360 |
| 40,000 | 3.80 | 3.65 | 18.36% | 399 |
| 50,000 | 3.62 | 3.45 | 24.82% | 475 |
| 60,000 | 3.49 | 3.42 | 24.78% | 467 |
| 70,000 | 3.48 | 3.35 | 24.42% | 457 |
| 80,000 | 3.47 | 3.25 | 25.09% | 465 |
| 90,000 | 3.46 | 3.25 | 22.59% | 478 |
| 100,000 | 3.53 | 3.26 | 22.71% | 482 |

**Translation quality at 30K steps** (improved sampling with proper target length estimation):
- "The weather is nice today." → "Die Wetter sind heute schön und gut." ✓
- "I have a cat." → "Ich habe eine Schütte." (structure OK, wrong noun)
- "The European Parliament has approved the proposal." → "Das Europäische Parlament hat den Vorschlag dafür zugestimmt." ✓✓
- "She went to the store to buy some milk." → "Sie kam in den Geschäften, um einige Milch zu kaufen." ✓
- "Hello, how are you?" → "Ja,, wie sind Sie in der Lage?" (weak)

**Translation quality at 50K steps**:
- "The weather is nice today." → "Das Wetter ist heute schön und gut." ✓✓ (fixed article)
- "I have a cat." → "Ich habe eine graue Katze." ✓✓ (correct noun! added "graue"=gray)
- "The European Parliament has approved the proposal." → "Das Europäische Parlament hat den Vorschlag auch verabschiedet." ✓✓
- "She went to the store to buy some milk." → "Sie kam in den Geschäft, um einige Menge Milch zu kaufen." ✓ (minor gender error)
- "Hello, how are you?" → "Hallo, wie sind Sie in der Zeit?" (partial)

**Sampling improvements**: Changed `p_sample_loop` to:
1. Estimate target length from source length (×1.5)
2. Initialize target as `[CLS] [MASK]... [SEP] [PAD]...` instead of all-MASK
3. Fix CLS/SEP/PAD positions during denoising (only unmask middle positions)
4. Per-sample mask count based on actual generatable positions

**Why it works**: Frozen pretrained embeddings break the tied-embedding feedback loop that
caused collapse in v13-v17. The pretrained BERT encoder provides strong initialization,
and the untied bottleneck head (768→256→vocab) prevents the output from destabilizing inputs.

**Final results (100K steps)**:
- Val loss: 3.26 (best 3.20 at 97.5K)
- Per-timestep accuracy: 91% at t=1, 26% at t=200
- Infilling accuracy: 51.5% on middle 20% of target
- Model produces legitimate EN→DE translations and infilling
- No collapse throughout entire training run

**Translation quality at 100K steps**:
- "The weather is nice today." → "Das Wetter ist heute schön und gut." ✓✓
- "I have a cat." → "Ich habe eine Katze zu haben." ✓ (correct noun, redundant verb)
- "The European Parliament has approved the proposal." → "Das Europäische Parlament hat den Vorschlag auch verabschiedet." ✓✓
- "She went to the store to buy some milk." → "Sie ging in den Geschäftsbereich, um einige Milch zu kaufen." ✓
- "Hello, how are you?" → "Hallo, wie sind Sie in der Lage?" (partial)

**Infilling at 100K steps** (5 positions per blank):
- "Sie ging ___ um Milch zu kaufen." → "Sie ging in das Geschäft, um Milch zu kaufen." ✓✓ (perfect!)
- "Das Wetter ist ___" → "Das Wetter ist heute schön." ✓✓ (perfect!)

**Status**: COMPLETE — first working discrete diffusion translation model.
Checkpoints: `checkpoints_v18_pretrained/model_final.pt` (and model_step_*.pt)

## Infrastructure (applies to both phases)

### Bug Fixes Applied (cumulative, continuous phase)
1. **emb_scale** (v5): `weight.norm(dim=-1).mean()` -> `weight.std()` (~32 -> ~1.0)
2. **x0 prediction** (v6): changed loss from epsilon to x0 target
3. **Cosine schedule** (v7): `alpha_bar = cos(pi/2 * t/T)^2`
4. **Min-SNR weighting** (v7): `w(t) = min(SNR(t), gamma) / SNR(t)`, gamma=5.0
5. **/embed_dim normalization** (v7): divide per-sample MSE by embed_dim
6. **LR warmup** (v8): linear warmup over 2000 optimizer steps
7. **Output LayerNorm** (v12): nn.LayerNorm before output_proj

### Code Features
- `train.py --resume <ckpt>`: checkpoint resumption with optimizer state
- Auto collapse detection: health checks at steps 500/1K/2K/5K + every val_every
- `eval.py --checkpoint <ckpt>`: per-timestep accuracy, translation, infilling eval
- Structured metrics in `<checkpoint_dir>/metrics.jsonl`

## Data
- **WMT14 EN→DE**, 4.5M training pairs
- Tokenized versions:
  - `data/wmt14_en_de_tokenized[_test]` — bert-base-multilingual-cased (120K vocab)
  - `data/wmt14_en_de_bert_cased[_test]` — bert-base-cased (29K vocab)
