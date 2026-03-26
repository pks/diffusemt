from dataclasses import dataclass


@dataclass
class Config:
    # Model — v32: deeper 512d (32 layers), back from 768d which underperformed
    pretrained_name: str = "bert-base-multilingual-cased"
    vocab_size: int = 119547
    embed_dim: int = 768       # embedding dimension (mBERT)
    model_dim: int = 512       # v32: back to 512d (v26 baseline width)
    num_heads: int = 8         # v32: 8 heads × 64d = 512
    architecture: str = "encoder-only"  # "encoder-decoder" or "encoder-only"
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_layers: int = 32       # v32: deeper (was 24 in v26)
    ff_dim: int = 2048         # v32: 4× model_dim
    dropout: float = 0.1
    max_seq_len: int = 128
    freeze_embeddings: bool = True
    self_cond: bool = False

    # Diffusion
    timesteps: int = 200
    schedule: str = "cosine"
    mask_token_id: int = 103  # [MASK] for bert-base-multilingual-cased
    diffusion_type: str = "mask"  # "source" (English-as-noise) or "mask" ([MASK]-based)

    # Training — v32: 512d×32L fits batch 40/GPU on TITAN RTX 24GB
    batch_size: int = 40
    grad_accum_steps: int = 13 # effective batch = 40 × 2 GPUs × 13 = 1040
    lr: float = 1e-4           # standard peak LR
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    num_train_steps: int = 200000
    log_every: int = 50
    val_every: int = 2500
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints_v32"
    grad_clip: float = 0.5

    # Loss weighting: uniform (v27b's 0.1 didn't improve BLEU over uniform)
    uncorrupted_loss_weight: float = 1.0

    # Length predictor (disabled — hurt BLEU in v27b)
    length_loss_weight: float = 0.0

    # Diffusion-only training (no AR warmup phase)
    ar_steps: int = 0                   # no autoregressive warmup
    curriculum_end_step: int = 0         # full t_max=200 from step 0
    curriculum_t_start: int = 1         # start curriculum at t_max=1

    # Auxiliary losses (legacy, unused)
    aux_mlm_weight: float = 0.0
    diversity_weight: float = 0.0

    # Data
    tokenizer_name: str = "bert-base-multilingual-cased"
    dataset_name: str = "wmt14"
    dataset_config: str = "de-en"
    src_lang: str = "en"
    tgt_lang: str = "de"

    # Device
    device: str = "cuda"
