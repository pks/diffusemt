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
    num_layers: int = 12  # for encoder-only architecture
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 128
    freeze_embeddings: bool = True

    # Diffusion
    timesteps: int = 200
    schedule: str = "cosine"
    mask_token_id: int = 103  # [MASK] for bert-base-multilingual-cased (fallback noise)

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 8
    lr: float = 1e-4
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    num_train_steps: int = 200000
    log_every: int = 50
    val_every: int = 2500
    save_every: int = 10000
    checkpoint_dir: str = "checkpoints_v24_encoder_only"
    grad_clip: float = 0.5

    # Diffusion-only training (no AR warmup phase)
    ar_steps: int = 0                   # no autoregressive warmup
    curriculum_end_step: int = 50000    # t_max reaches T at this step
    curriculum_t_start: int = 1         # start curriculum at t_max=1

    # Auxiliary losses
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
