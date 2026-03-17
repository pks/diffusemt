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

---

## Phase 3: Source-as-Corruption Discrete Diffusion (v19-v20)

**Approach B**: Forward diffusion replaces German target tokens with English source tokens.
At t=T the sequence IS English, at t=0 it's German. Translation = reverse diffusion from English.
Where source is shorter than target, [MASK] is fallback noise. Single sequence (no concat).

Motivation: v18's infilling was confounded by BERT's native MLM capability for same-language
German. Source-as-corruption makes infilling genuinely cross-lingual — the "noise" is English
tokens, which BERT's MLM can't denoise.

### v19: Source-as-Corruption with Untied Bottleneck Head — COLLAPSED

**Architecture**: Pretrained mBERT encoder + from-scratch untied bottleneck output head
(LayerNorm → Linear(768,256) → GELU → Linear(256,vocab)). Single sequence, no segment
embeddings. Timestep conditioning. Loss on corrupted positions only.

**Result**: Collapsed at step 1000. Unique tokens dropped from 499 (step 500) to 3 (step 1000).
The randomly-initialized output head doesn't get enough gradient signal from the sparse
corrupted-position loss, and collapses to predicting a few common tokens.

### v19b: x0-Loss + Tied Projection — COLLAPSED

Changed to x0-parameterization (loss on ALL target positions, not just corrupted) and tied
output projection (h @ embedding_weight.T). Still collapsed at step 1000 (unique=1 at random
init tied projection, unique=5 at pretrained-LN tied projection). The tied projection delays
but doesn't prevent collapse when the transform layers are randomly initialized.

### v20: MLM-Head-Initialized Output + x0-Loss — SOLVED

**Key insight**: Initialize the output head transform from BERT's pretrained MLM head
(dense + LayerNorm + bias), not from scratch. The pretrained init produces diverse outputs
from the start, preventing the collapse attractor.

**Architecture**:
- Pretrained mBERT encoder + encoder layers (same as v18)
- Frozen word embeddings (91.8M frozen)
- Output head: `output_proj` (pretrained from BERT MLM dense), `output_norm` (pretrained from BERT MLM LayerNorm), tied projection via `h @ word_embedding.T + output_bias`
- Sinusoidal timestep conditioning
- x0-parameterization: loss on ALL real target positions
- 179.2M total params, 87.3M trainable
- Single sequence (no concat, no segment embeddings)

**Training config**:
- DDP on 2x TITAN RTX, batch_size=96/GPU, grad_accum=6 → effective batch 1152
- lr=3e-5, warmup=4000 optimizer steps (24K training steps), cosine decay to 100K
- Mixed precision (fp16), AdamW, grad clip 1.0
- T=200 timesteps, cosine schedule

| Step | Train Loss | Val Loss | t=100 acc | t=100 unique | t=200 acc | t=200 unique |
|------|-----------|----------|-----------|--------------|-----------|--------------|
| 500 | 22.32 | - | 2.48% | 235 | 1.34% | 416 |
| 1,000 | 7.01 | - | 7.38% | 371 | 4.75% | 403 |
| 2,000 | 5.00 | - | 21.03% | 600 | 10.62% | 308 |
| 2,500 | 4.67 | 3.21 | 26.83% | 583 | 13.09% | 299 |
| 5,000 | 4.20 | 2.79 | 31.70% | 603 | 14.94% | 258 |
| 10,000 | 4.08 | 2.59 | 37.83% | 610 | 15.74% | 308 |
| 20,000 | 3.87 | 2.43 | 42.33% | 646 | 16.57% | 398 |
| 50,000 | 3.57 | 2.20 | 45.60% | 744 | 18.11% | 502 |
| 75,000 | 3.51 | 2.10 | 48.47% | 751 | 18.35% | 606 |
| 100,000 | 3.52 | 2.17 | 51.92% | 639 | 22.39% | 567 |

**High initial loss (35.5)**: The pretrained MLM head confidently predicts wrong tokens for the
cross-lingual denoising task. This rapidly drops as the model adapts (35 → 7 in first 1000 steps).

**Translation quality at 10K steps**:
- "Das Europäische Parlament hat die Vorschlagen bewändet." (forming German structure)
- "Ich habe einen cat - Hund gefunden." (English still leaking)

**Translation quality at 50K steps**:
- "Das Europäische Parlament hat den Vorschlag zu dieser Frage vorgelegt." (good grammar)
- "Sie kamen zum Coffee Shop, um einige Verkaufsreise zu kaufen." (knows "zu kaufen")

**Final results (100K steps)**:
- Val loss: 2.17 (best 2.10 at 75K)
- Per-timestep accuracy: 100% at t=1, 66% at t=81, 58% at t=101, 25% at t=200
- Infilling accuracy: 27.18% on middle 20%, 100% on known positions

**Translation quality at 100K steps**:
- "The European Parliament has approved the proposal." → "Das Europäische Parlament hat den Vorschlag der Kommission gestimmt."
- "She went to the store to buy some milk." → "Sie kamen zum Coffee Shop, um eine Menge von Milch zu kaufen."
- "I have a cat." → "Ich habe einen Schlüssel." (wrong noun but correct case/article)

**Infilling at 100K steps**:
- "Ich habe ___ Katze." → "Ich habe eine Katze, eine Katze." (correct article)
- "Sie ging ___ um Milch zu kaufen." → "Sie ging in den Großhandel, um Milch zu kaufen." ✓
- "___ eine Katze." → "Ich habe a.. eine Katze." (reconstructs subject)

**Why v20 works where v19 failed**: The randomly-initialized output head collapses because it
starts in a low-diversity region of parameter space and the gradients (even with x0-loss) are
insufficient to escape. The pretrained MLM head provides a high-diversity starting point —
it already produces diverse token predictions, and the model only needs to adapt those predictions
from same-language MLM to cross-lingual denoising.

**Comparison with v18**:
- v18 (mask-based, concat): val_loss 3.26, infill accuracy 51.5%
- v20 (source-as-corruption): val_loss 2.17, infill accuracy 27.2%
- v20 has better val loss but lower infill accuracy (task is harder: cross-lingual noise)
- v20's infilling is genuinely cross-lingual (not confounded by BERT's same-language MLM)

**Status**: COMPLETE. Checkpoints: `checkpoints_v20_sourcecorrupt/model_final.pt`

---

## Phase 4: From-Scratch Encoder-Decoder (v21-v23)

**Goal**: No pretrained model (only frozen mBERT embeddings). From-scratch Pre-LN encoder-decoder with source-as-corruption diffusion.

### v21: Two-Phase Training (AR Warmup → Diffusion) — SOLVED FROM-SCRATCH COLLAPSE

**Architecture**: From-scratch Pre-LN encoder-decoder
- 6 encoder layers (PreLNEncoderLayer: self-attn + FF)
- 6 decoder layers (PreLNDecoderLayer: self-attn + cross-attn + FF)
- model_dim=512, num_heads=8, ff_dim=2048, embed_dim=768 (mBERT)
- Frozen mBERT word embeddings (input) + frozen output_embedding (registered buffer)
- embed_proj: Linear(768→512), learned position embeddings, sinusoidal timestep conditioning
- Output: output_proj(h) @ output_embedding.T + output_bias
- Auxiliary encoder MLM head (15% masking, weight=0.1)
- 138.1M total, 46.3M trainable, 91.8M frozen

**Key innovation**: Two-phase training
- Phase 1 (steps 0-30K): Autoregressive seq2seq with causal masking + teacher forcing
- Phase 2 (steps 30K-100K): Source-as-corruption diffusion with timestep curriculum (t_max ramps 10→200 from step 30K to 50K)

**Why two-phase works**: From-scratch encoder collapses in diffusion because it produces degenerate representations. AR warmup establishes stable encoder-decoder cross-attention patterns that survive the transition to diffusion.

**Collapse attempts 1-4 failed**: Frozen/trainable output embedding, reconstruction warmup, skip connections — all collapsed at step 2000-2500.

**Training**: batch_size=64, grad_accum=8, lr=1e-4, warmup=8000 opt steps, 2x TITAN RTX DDP

| Step | Val Loss | t=100 acc | t=100 unique | t=200 acc | t=200 unique |
|------|----------|-----------|--------------|-----------|--------------|
| 32500 | 5.07 | 7.4% | 484 | 6.1% | 891 |
| 50000 | 2.82 | 27.9% | 330 | 17.9% | 319 |
| 70000 | 2.33 | 39.3% | 454 | 19.7% | 391 |
| 90000 | 2.21 | 43.1% | 486 | 18.3% | 379 |

**BLEU (90K, temp=1.5, ratio=1.2)**: 9.40
**BLEU (90K, temp=2.0, ratio=1.1)**: ~9.3
Process crashed with SIGABRT during final save at 100K (OOM on val+health+save), but 90K checkpoint has essentially converged (LR was near zero).

### v22: Extended Training (200K total) — LR Restart

Resumed from v21 90K checkpoint with num_train_steps=200K (fresh cosine schedule). The LR restart from near-zero to ~9e-05 gave continued learning.

| Step | Val Loss | t=100 acc | t=200 acc |
|------|----------|-----------|-----------|
| 100K | 2.11 | 45.9% | 18.3% |
| 122.5K | 1.96 | 47.5% | 18.6% |
| 137.5K | 1.91 | — | — |
| 160K | 1.87 | ~53% | ~24% |
| 180K | 1.83 (best) | ~55% | ~24% |
| 190K | 1.97 | ~55% | ~25% |

**BLEU results (190K checkpoint)**:

| Temperature | Length Ratio | BLEU |
|-------------|-------------|------|
| 1.0 | 1.1 | 11.92 |
| 1.5 | 1.2 | 12.49 |
| 2.0 | 1.1 | **13.30** |
| 2.0 | 1.2 | 11.96 |
| 3.0 | 1.1 | 12.98 |

**Best BLEU: 13.30** (temp=2.0, ratio=1.1)

**Infilling accuracy**: 40.8% on middle 20% of target (up from 27.2% in v20)

**Per-timestep accuracy (190K)**:
- t=1: 100%, t=81: 73%, t=101: 63%, t=141: 43%, t=200: 26%

**Translation quality (190K)**:
- "The European Parliament has approved the proposal." → "Das Europäische Parlament hat den Vorschlag verabschiedet." ✓✓
- "The weather is nice today." → "Das Wetter ist heute schönlich gewesen." ✓ (grammatically odd)
- "She went to the store to buy some milk." → "Sie ging nach dem Geschäftswerk, einige Milch zu kaufen." ✓

**Sampling improvements**:
- Temperature scaling during confidence-based unmasking (higher temp = softer confidence → better re-ranking)
- Step skipping: 100 steps gives 13.22 vs 200 steps 13.30 (half the compute, 99.4% quality)
- Length ratio 1.1 beats 1.2 (brevity penalty vs padding noise tradeoff)

Process crashed with SIGABRT during final save at 200K (same OOM as v21).

### v23: Extended Training (300K total) — Second LR Restart

Resumed from v22 190K with num_train_steps=300K. Fresh cosine schedule, LR starts at ~4.5e-05.
Currently training.

**Status**: IN PROGRESS. Checkpoints: `checkpoints_v23_extended/`

**Comparison with previous phases**:
| Version | Model | Pretrained | Val Loss | BLEU | Infill Acc |
|---------|-------|------------|----------|------|------------|
| v18 | mBERT encoder-only | Full | 3.26 | — | 51.5% |
| v20 | mBERT encoder-only | Full | 2.17 | 7.59 | 27.2% |
| v21 | From-scratch enc-dec | Embeddings only | 2.21 | 9.40 | — |
| v22 | From-scratch enc-dec | Embeddings only | 1.83 | **13.30** | 40.8% |

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

## Phase 5: Encoder-Only Deep Diffusion (v24+)

### v24 — Deep encoder-only, diffusion-only (no AR warmup)
- **Architecture**: `SourceCorruptionEncoderOnly` — 12-layer Pre-LN self-attention
  - Concatenates [source | corrupted_target] (256 tokens total)
  - Segment embeddings (0=source, 1=target), position embeddings for 2×max_seq_len
  - Timestep conditioning added to all positions
  - 131.9M params total, 40.1M trainable, 91.8M frozen (mBERT embeddings)
- **Training**: Pure diffusion from step 0 (no AR warmup phase)
  - Curriculum: t_max ramps 1→200 over first 50K steps (gentle start to avoid collapse)
  - LR warmup: 2K steps (fast ramp to help model learn at low timesteps)
  - batch_size=64, grad_accum=8, effective_batch=1024 (2× TITAN RTX 24GB)
  - GPU utilization: 22.2GB/24.6GB per GPU (90%)
  - 200K total steps, cosine LR with 2K warmup
- **Status**: Training complete (200K steps)
- **Motivation**: User asked to make encoder more powerful — single deep encoder
  sees full [source|target] context instead of split enc/dec.
- **Results**:
  - val_loss trajectory: 8.02 → 4.73 → 2.14 → 1.95 → 1.88 → 1.78 → **1.77** (197.5K best)
  - BLEU 80K (50 steps, temp=2.0): 11.06
  - BLEU 100K (50 steps, temp=2.0): 11.58
  - BLEU 190K (50 steps, temp=2.0): **12.73** (best)
  - BLEU 190K (200 steps, temp=2.0): 12.51
  - BLEU 190K (200 steps, temp=1.5): 12.31
  - BLEU 190K (50 steps, temp=1.0): 12.32
  - For comparison: v22 enc-dec 190K BLEU=13.30 (200 steps), val_loss=1.83
  - t=100 acc=63%, t=200 acc=22% at 195K (healthy)
  - **Does NOT beat v22 enc-dec** (12.73 vs 13.30)
  - More diffusion steps (200 vs 50) did not help — possibly because the source-as-noise
    scheme doesn't benefit from fine-grained iterative refinement the same way mask diffusion would
  - Infilling works but with grammatical errors and English leakage
  - Sample translations show reasonable German output with occasional English words
- **Key findings**:
  - Diffusion-only (no AR warmup) works if curriculum starts at t_max=1 and ramps slowly
  - Initial attempts with t_max=10 start caused collapse at step 2K
  - Fast LR warmup (2K steps) helps model learn at low timesteps
  - Health check must test within current curriculum range to avoid false collapse alarms
  - DDP requires `find_unused_parameters=True` (aux_mlm_head unused when weight=0)
  - Always maximize GPU memory — test batch sizes to find max that fits
  - Encoder-only with source-as-noise doesn't clearly beat enc-dec (v22) — suggests the
    architecture bottleneck isn't the main issue; the corruption strategy may need rethinking
  - 50 sampling steps slightly outperformed 200 steps — unusual, may indicate the reverse
    process is miscalibrated for this corruption schedule

## Data
- **WMT14 EN→DE**, 4.5M training pairs
- Tokenized versions:
  - `data/wmt14_en_de_tokenized[_test]` — bert-base-multilingual-cased (120K vocab)
  - `data/wmt14_en_de_bert_cased[_test]` — bert-base-cased (29K vocab)
