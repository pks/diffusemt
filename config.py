from dataclasses import dataclass


@dataclass
class Config:
    # Model — v34: best model (21.18 BLEU), 40-layer encoder-only, mask diffusion
    pretrained_name: str = "bert-base-multilingual-cased"
    vocab_size: int = 119547
    embed_dim: int = 768       # embedding dimension (mBERT)
    model_dim: int = 512
    num_heads: int = 8         # 8 heads × 64d = 512
    architecture: str = "encoder-only"  # "encoder-decoder" or "encoder-only"
    encoder_layers: int = 6
    decoder_layers: int = 6
    num_layers: int = 40       # v34: 40 layers (best BLEU)
    ff_dim: int = 2048         # 4× model_dim
    dropout: float = 0.1
    max_seq_len: int = 128
    freeze_embeddings: bool = False  # v33+: unfreeze for fine-tuning
    embed_lr: float = 1e-5           # v33+: low LR for pretrained embeddings
    self_cond: bool = False

    # Diffusion
    timesteps: int = 200
    schedule: str = "cosine"
    mask_token_id: int = 103  # [MASK] for bert-base-multilingual-cased
    diffusion_type: str = "mask"  # "source" (English-as-noise) or "mask" ([MASK]-based)

    # Classifier-free guidance: disabled (v35 showed CFG hurts for discrete diffusion)
    cfg_dropout: float = 0.0

    # Training — v34: 40 layers, unfrozen embeddings
    batch_size: int = 36
    grad_accum_steps: int = 15 # effective batch = 36 × 2 GPUs × 15 = 1080
    lr: float = 1e-4
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    num_train_steps: int = 200000
    log_every: int = 50
    val_every: int = 2500
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints_v34"
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
