from dataclasses import dataclass


@dataclass
class Config:
    # Model (512d/8L — proven stable with source-init)
    embed_dim: int = 512
    num_heads: int = 8
    num_layers: int = 8
    ff_dim: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 128

    # Diffusion
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "cosine"
    min_snr_gamma: float = 5.0

    # Training
    batch_size: int = 8192
    grad_accum_steps: int = 1
    lr: float = 1e-4
    warmup_steps: int = 1000
    num_train_steps: int = 50000
    log_every: int = 50
    val_every: int = 2500
    save_every: int = 2500
    checkpoint_dir: str = "checkpoints_v23"
    ce_loss_weight: float = 0.1

    # Data
    tokenizer_name: str = "bert-base-multilingual-cased"
    dataset_name: str = "wmt14"
    dataset_config: str = "de-en"
    src_lang: str = "en"
    tgt_lang: str = "de"

    # Device
    device: str = "cuda"
