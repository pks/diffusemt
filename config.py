from dataclasses import dataclass


@dataclass
class Config:
    # Model — deep encoder-only with frozen mBERT embeddings
    pretrained_name: str = "bert-base-multilingual-cased"
    vocab_size: int = 119547
    embed_dim: int = 768       # embedding dimension (mBERT)
    model_dim: int = 512       # encoder/decoder hidden dimension
    num_heads: int = 8
    architecture: str = "encoder-only"  # "encoder-decoder" or "encoder-only"
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_layers: int = 24  # for encoder-only architecture (v26: doubled from v25's 12)
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 128
    freeze_embeddings: bool = True

    # Diffusion
    timesteps: int = 200
    schedule: str = "cosine"
    mask_token_id: int = 103  # [MASK] for bert-base-multilingual-cased
    diffusion_type: str = "mask"  # "source" (English-as-noise) or "mask" ([MASK]-based)

    # Training
    batch_size: int = 48      # 24-layer model on TITAN RTX 24GB (single-GPU peak 19.6 GB; DDP adds ~2 GB)
    grad_accum_steps: int = 11  # effective batch = 48 × 2 GPUs × 11 = 1056 ≈ v25's 1024
    lr: float = 2e-5          # v27b: gentler LR for fine-tuning (5e-5 was too aggressive)
    warmup_steps: int = 1000
    label_smoothing: float = 0.1
    num_train_steps: int = 100000
    log_every: int = 50
    val_every: int = 2500
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints_v27b"
    grad_clip: float = 0.5

    # Loss weighting (v27): upweight masked positions, downweight uncorrupted
    uncorrupted_loss_weight: float = 0.1

    # Length predictor (v27)
    length_loss_weight: float = 0.2

    # Diffusion-only training (no AR warmup phase)
    ar_steps: int = 0                   # no autoregressive warmup
    curriculum_end_step: int = 0        # v26: start at full t_max=200 (init from trained weights)
    curriculum_t_start: int = 1         # start curriculum at t_max=1

    # Auxiliary losses (legacy, unused in v27)
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
