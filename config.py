from dataclasses import dataclass


@dataclass
class Config:
    # Model
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 128  # per side (source or target)

    # Diffusion
    timesteps: int = 200
    schedule: str = "cosine"
    mask_token_id: int = 103  # [MASK] for bert-base-multilingual-cased

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 6
    lr: float = 3e-5
    warmup_steps: int = 2000
    label_smoothing: float = 0.1
    num_train_steps: int = 50000
    log_every: int = 50
    val_every: int = 2500
    save_every: int = 10000
    checkpoint_dir: str = "checkpoints_v17_smallvocab"

    # Data
    tokenizer_name: str = "bert-base-cased"
    dataset_name: str = "wmt14"
    dataset_config: str = "de-en"
    src_lang: str = "en"
    tgt_lang: str = "de"

    # Device
    device: str = "cuda"
