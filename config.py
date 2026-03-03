from dataclasses import dataclass


@dataclass
class Config:
    # Model (BERT-base sized)
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    ff_dim: int = 3072
    dropout: float = 0.1
    max_seq_len: int = 128

    # Diffusion
    timesteps: int = 200
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Training
    batch_size: int = 64
    grad_accum_steps: int = 4
    lr: float = 1e-4
    num_train_steps: int = 50000
    log_every: int = 100
    val_every: int = 2500
    save_every: int = 5000
    checkpoint_dir: str = "checkpoints"

    # Data
    tokenizer_name: str = "bert-base-multilingual-cased"
    dataset_name: str = "wmt14"
    dataset_config: str = "de-en"
    src_lang: str = "en"
    tgt_lang: str = "de"

    # Device
    device: str = "cuda"
