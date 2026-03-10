# GPU Memory Profiling Results

## Single-GPU Profiling (no DDP overhead)

### Batch size sweep — BERT-base (294M) without checkpointing
| Batch | Peak MB | % Used | samp/s |
|-------|---------|--------|--------|
| 96    | 15,849  | 66.0%  | 30     |
| 128   | 19,870  | 82.7%  | 30     |
| 160   | OOM     | -      | -      |

### Batch size sweep — BERT-base (294M) with checkpointing
| Batch | Peak MB | % Used | samp/s |
|-------|---------|--------|--------|
| 128   | 13,354  | 55.6%  | 24     |
| 192   | 18,221  | 75.8%  | 24     |
| 224   | 20,656  | 86.0%  | 24     |
| 256   | 23,102  | 96.2%  | 24     |

### Model scaling (single-GPU, with checkpointing, bs=64)
| Config                  | Params | Train MB | Inf bs=1 | Verdict   |
|-------------------------|--------|----------|----------|-----------|
| 768d 12L (BERT-base)    | 294M   | 8,488    | 1,521    | OK        |
| 1024d 12L               | 481M   | 12,363   | 2,474    | OK        |
| 1024d 16L               | 599M   | 15,859   | 3,123    | OK        |
| 1024d 20L               | 716M   | 19,354   | 3,771    | OK        |
| 1024d 24L (BERT-large)  | 834M   | 22,849   | 4,420    | TRAIN OOM |

### BERT-large batch sweep (single-GPU, with checkpointing)
| Batch | No ckpt    | With ckpt  |
|-------|-----------|------------|
| 32    | 21,419 MB | 16,607 MB  |
| 48    | OOM       | 19,744 MB  |
| 56    | OOM       | 21,323 MB  |

## DDP Reality (2x GPU)
- DDP adds ~6 GiB for 834M param model (gradient buckets + comm buffers)
- Whole-encoder/decoder checkpointing: OOM at bs=40 (22.9 GiB allocated)
- Per-layer checkpointing: works at bs=48 (~training confirmed stable)
- Profiling scripts: profile_memory.py, profile_model_size.py, profile_bert_large.py, profile_ckpt_compare.py
