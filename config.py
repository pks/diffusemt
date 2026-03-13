from dataclasses import dataclass


@dataclass
class Config:
    # Model (matches bert-base-multilingual-cased)
    pretrained_name: str = "bert-base-multilingual-cased"
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ff_dim: int = 3072
    bottleneck_dim: int = 256
    dropout: float = 0.1
    max_seq_len: int = 128  # per side (source or target)
    freeze_embeddings: bool = True

    # Diffusion
    timesteps: int = 200
    schedule: str = "cosine"
    mask_token_id: int = 103  # [MASK] for bert-base-multilingual-cased

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 4
    lr: float = 5e-5
    warmup_steps: int = 2000
    label_smoothing: float = 0.0
    num_train_steps: int = 100000
    log_every: int = 50
    val_every: int = 2500
    save_every: int = 10000
    checkpoint_dir: str = "checkpoints_v18_pretrained"

    # Data
    tokenizer_name: str = "bert-base-multilingual-cased"
    dataset_name: str = "wmt14"
    dataset_config: str = "de-en"
    src_lang: str = "en"
    tgt_lang: str = "de"

    # Device
    device: str = "cuda"
