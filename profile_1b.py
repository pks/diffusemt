"""
Quick DDP-realistic profiler: measures actual memory with per-layer
gradient checkpointing to find configs that fit ~1B params on 2x24GB.

Simulates DDP overhead by adding gradient buffer memory.
"""

import gc
import torch
from config import Config
from model import DiffusionTransformer
from diffusion import GaussianDiffusion


def get_gpu_mem_mb():
    return torch.cuda.max_memory_allocated() / 1024**2

def reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def enable_per_layer_checkpointing(model):
    from torch.utils.checkpoint import checkpoint

    for layer in model.source_encoder.layers:
        orig = layer.forward
        def make(fn):
            def fwd(src, src_mask=None, src_key_padding_mask=None, is_causal=None):
                def f(s, m, kpm): return fn(s, src_mask=m, src_key_padding_mask=kpm)
                return checkpoint(f, src, src_mask, src_key_padding_mask, use_reentrant=False)
            return fwd
        layer.forward = make(orig)

    for layer in model.target_decoder.layers:
        orig = layer.forward
        def make(fn):
            def fwd(tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    tgt_is_causal=None, memory_is_causal=None):
                def f(t, m, mkpm): return fn(t, m, memory_key_padding_mask=mkpm)
                return checkpoint(f, tgt, memory, memory_key_padding_mask, use_reentrant=False)
            return fwd
        layer.forward = make(orig)


def profile_ddp_training(embed_dim, num_heads, num_layers, ff_dim, batch_size, device):
    """Profile with DDP-realistic memory (allocate fake gradient buffer)."""
    config = Config()
    reset_memory()

    model = DiffusionTransformer(
        vocab_size=120000, embed_dim=embed_dim, num_heads=num_heads,
        num_layers=num_layers, ff_dim=ff_dim, dropout=0.1,
        max_seq_len=config.max_seq_len,
    ).to(device)

    params = count_params(model)

    # Simulate DDP gradient buffer overhead (~4 bytes per param)
    ddp_buffer = torch.empty(params, dtype=torch.float32, device=device)

    enable_per_layer_checkpointing(model)

    with torch.no_grad():
        emb_scale = model.token_embedding.weight.norm(dim=-1).mean().item()

    diffusion = GaussianDiffusion(
        timesteps=config.timesteps, beta_start=config.beta_start,
        beta_end=config.beta_end,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    source_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    source_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)
    target_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    target_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)

    model.train()
    try:
        for i in range(3):
            with torch.no_grad():
                x0 = model.token_embedding(target_ids) / emb_scale
            B = x0.shape[0]
            t = torch.randint(0, config.timesteps, (B,), device=device)
            xt, noise = diffusion.q_sample(x0, t)

            x0_self_cond = None
            if i % 2 == 0:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    pred_eps = model(source_ids, source_mask, xt, t)
                    x0_self_cond = diffusion.predict_x0_from_eps(xt, t, pred_eps).detach()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                predicted_eps = model(source_ids, source_mask, xt, t, x0_self_cond=x0_self_cond)

            mask = target_mask.unsqueeze(-1).float()
            loss = ((predicted_eps.float() - noise) ** 2 * mask).sum() / mask.sum() / embed_dim
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        peak = get_gpu_mem_mb()

        del optimizer, scaler, ddp_buffer, model, diffusion
        del source_ids, source_mask, target_ids, target_mask
        reset_memory()
        return params, peak

    except torch.cuda.OutOfMemoryError:
        del optimizer, scaler, ddp_buffer
        reset_memory()
        return params, -1


def main():
    device = torch.device("cuda:0")
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_total:.0f} MB)")
    print(f"Profiling with simulated DDP buffer + per-layer checkpointing")
    print()

    # Configs targeting ~1B params
    configs = [
        # (embed, heads, layers, ff, label)
        (1024, 16, 24, 4096,  "current: 1024d 24L"),
        (1024, 16, 28, 4096,  "1024d 28L"),
        (1024, 16, 32, 4096,  "1024d 32L"),
        (1152, 18, 24, 4608,  "1152d 24L"),
        (1152, 18, 28, 4608,  "1152d 28L"),
        (1280, 20, 24, 5120,  "1280d 24L"),
        (1280, 20, 20, 5120,  "1280d 20L"),
        (1024, 16, 36, 4096,  "1024d 36L"),
    ]

    batch_size = 48  # test with the known-working batch size

    print(f"{'Config':<22} | {'Params':>8} | {'Peak MB':>9} | {'% GPU':>7} | {'Headroom':>9} | Status")
    print(f"{'-'*22}-+-{'-'*8}-+-{'-'*9}-+-{'-'*7}-+-{'-'*9}-+-------")

    for embed_dim, num_heads, num_layers, ff_dim, label in configs:
        params, peak = profile_ddp_training(embed_dim, num_heads, num_layers, ff_dim, batch_size, device)
        params_m = params / 1e6

        if peak < 0:
            print(f"{label:<22} | {params_m:>6.0f}M | {'OOM':>9} | {'---':>7} | {'---':>9} | OOM")
        else:
            pct = peak / gpu_total * 100
            headroom = gpu_total - peak
            status = "OK" if pct < 90 else ("TIGHT" if pct < 95 else "DANGER")
            print(f"{label:<22} | {params_m:>6.0f}M | {peak:>8.0f} | {pct:>6.1f}% | {headroom:>7.0f} MB | {status}")


if __name__ == "__main__":
    main()
