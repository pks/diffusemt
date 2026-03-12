"""
GPU memory profiler for diffusemt.

Tests increasing batch sizes on a single GPU to find the maximum
that fits in memory. Reports peak memory, throughput, and whether
gradient checkpointing helps.
"""

import gc
import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from config import Config
from model import DiffusionTransformer
from diffusion import GaussianDiffusion


def get_gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024**2


def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def enable_gradient_checkpointing(model):
    """Enable gradient checkpointing on encoder and decoder."""
    if hasattr(model.source_encoder, 'layers'):
        for layer in model.source_encoder.layers:
            layer.self_attn._qkv_same_embed_dim = True  # needed for checkpoint compat
    if hasattr(model.target_decoder, 'layers'):
        pass

    # Wrap encoder and decoder with checkpointing
    original_encoder_forward = model.source_encoder.forward
    original_decoder_forward = model.target_decoder.forward

    def checkpointed_encoder_forward(src, mask=None, src_key_padding_mask=None, is_causal=None):
        def custom_forward(src_, mask_, src_key_padding_mask_):
            return original_encoder_forward(
                src_, mask=mask_, src_key_padding_mask=src_key_padding_mask_
            )
        return torch.utils.checkpoint.checkpoint(
            custom_forward, src, mask, src_key_padding_mask,
            use_reentrant=False
        )

    def checkpointed_decoder_forward(tgt, memory, tgt_mask=None, memory_mask=None,
                                      tgt_key_padding_mask=None,
                                      memory_key_padding_mask=None,
                                      tgt_is_causal=None, memory_is_causal=None):
        def custom_forward(tgt_, memory_, memory_key_padding_mask_):
            return original_decoder_forward(
                tgt_, memory_, memory_key_padding_mask=memory_key_padding_mask_
            )
        return torch.utils.checkpoint.checkpoint(
            custom_forward, tgt, memory, memory_key_padding_mask,
            use_reentrant=False
        )

    model.source_encoder.forward = checkpointed_encoder_forward
    model.target_decoder.forward = checkpointed_decoder_forward
    return model


def profile_batch_size(batch_size, config, model, diffusion, emb_scale, device,
                       grad_accum_steps=1, warmup=2, measure=3):
    """Run a few training steps at given batch_size, return peak memory and step time."""
    reset_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.amp.GradScaler("cuda")

    # Synthetic data
    source_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    source_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)
    target_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    target_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)

    model.train()
    total_steps = warmup + measure
    step_times = []

    try:
        for i in range(total_steps):
            t0 = time.time()

            with torch.no_grad():
                x0 = model.token_embedding(target_ids) / emb_scale

            B = x0.shape[0]
            t = torch.randint(0, config.timesteps, (B,), device=device)
            xt, noise = diffusion.q_sample(x0, t)

            # Self-conditioning (50% of time, like real training)
            x0_self_cond = None
            if i % 2 == 0:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    pred_eps = model(source_ids, source_mask, xt, t)
                    x0_self_cond = diffusion.predict_x0_from_eps(xt, t, pred_eps).detach()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                predicted_eps = model(source_ids, source_mask, xt, t,
                                      x0_self_cond=x0_self_cond)

            mask = target_mask.unsqueeze(-1).float()
            loss = ((predicted_eps.float() - noise) ** 2 * mask).sum() / mask.sum() / config.embed_dim
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            elapsed = time.time() - t0

            if i >= warmup:
                step_times.append(elapsed)

        peak_mem = get_gpu_mem_mb()
        avg_time = sum(step_times) / len(step_times) if step_times else 0
        samples_per_sec = batch_size / avg_time if avg_time > 0 else 0

        # Clean up optimizer
        del optimizer, scaler, source_ids, source_mask, target_ids, target_mask
        reset_memory()

        return {
            "batch_size": batch_size,
            "peak_mem_mb": peak_mem,
            "avg_step_ms": avg_time * 1000,
            "samples_per_sec": samples_per_sec,
            "oom": False,
        }

    except torch.cuda.OutOfMemoryError:
        # Clean up after OOM
        del optimizer, scaler
        if 'source_ids' in dir():
            del source_ids, source_mask, target_ids, target_mask
        reset_memory()
        model.zero_grad(set_to_none=True)
        return {
            "batch_size": batch_size,
            "peak_mem_mb": -1,
            "avg_step_ms": -1,
            "samples_per_sec": -1,
            "oom": True,
        }


def main():
    device = torch.device("cuda:0")
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = tokenizer.vocab_size

    # GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {gpu_name}")
    print(f"Total memory: {gpu_mem_total:.0f} MB")
    print(f"Model: {config.embed_dim}d, {config.num_layers}L, {config.num_heads}H, ff={config.ff_dim}")
    print(f"Seq len: {config.max_seq_len}")
    print()

    # Batch sizes to test
    batch_sizes = [32, 64, 96, 128, 160, 192, 224, 256, 320, 384]

    for use_checkpointing in [False, True]:
        label = "WITH gradient checkpointing" if use_checkpointing else "WITHOUT gradient checkpointing"
        print(f"{'='*60}")
        print(f"  Profiling {label}")
        print(f"{'='*60}")
        print(f"{'Batch':>6} | {'Peak MB':>9} | {'% Used':>7} | {'ms/step':>8} | {'samp/s':>8} | Status")
        print(f"{'-'*6}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-------")

        # Build fresh model for each test
        model = DiffusionTransformer(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
        ).to(device)

        if use_checkpointing:
            model = enable_gradient_checkpointing(model)

        with torch.no_grad():
            emb_scale = model.token_embedding.weight.norm(dim=-1).mean().item()

        best_throughput = 0
        best_bs = 0

        for bs in batch_sizes:
            result = profile_batch_size(bs, config, model, diffusion=GaussianDiffusion(
                timesteps=config.timesteps,
                beta_start=config.beta_start,
                beta_end=config.beta_end,
            ).to(device), emb_scale=emb_scale, device=device, grad_accum_steps=1)

            if result["oom"]:
                print(f"{bs:>6} | {'OOM':>9} | {'---':>7} | {'---':>8} | {'---':>8} | OOM")
                break
            else:
                pct = result["peak_mem_mb"] / gpu_mem_total * 100
                print(f"{bs:>6} | {result['peak_mem_mb']:>8.0f} | {pct:>6.1f}% | {result['avg_step_ms']:>7.0f} | {result['samples_per_sec']:>7.0f} | OK")
                if result["samples_per_sec"] > best_throughput:
                    best_throughput = result["samples_per_sec"]
                    best_bs = bs

        print()
        print(f"  Best throughput: batch_size={best_bs} -> {best_throughput:.0f} samples/sec")
        print()

        del model
        reset_memory()

    # Recommendations
    print("=" * 60)
    print("  RECOMMENDATIONS")
    print("=" * 60)
    print()
    print("For 2x GPU DDP training, pick the largest batch_size that")
    print("stays under ~90% memory (leave room for variance).")
    print()
    print("Effective batch = batch_size × 2 GPUs × grad_accum_steps")
    print("Current: 96 × 2 × 2 = 384")
    print()
    print("If max batch_size is e.g. 192:")
    print("  -> Use batch_size=192, grad_accum=1 -> effective 384 (same, but faster)")
    print("  -> Or batch_size=192, grad_accum=2 -> effective 768 (larger effective batch)")


if __name__ == "__main__":
    main()
