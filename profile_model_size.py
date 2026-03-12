"""
Profile how large we can make the model given:
  - Training: 2x 24GB GPUs with DDP + gradient checkpointing
  - Inference: 1x 24GB GPU (the binding constraint)

Tests various model configs (embed_dim, num_layers, ff_dim) and reports
peak memory for both training and inference.
"""

import gc
import time
import torch
import torch.nn as nn
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


def enable_gradient_checkpointing(model):
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


def profile_training(model, config, diffusion, emb_scale, device, batch_size):
    """Simulate one training step with self-conditioning + grad checkpointing."""
    reset_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    source_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    source_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)
    target_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    target_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)

    model.train()
    try:
        # Warmup
        for i in range(2):
            with torch.no_grad():
                x0 = model.token_embedding(target_ids) / emb_scale
            B = x0.shape[0]
            t = torch.randint(0, config.timesteps, (B,), device=device)
            xt, noise = diffusion.q_sample(x0, t)

            # Self-conditioning pass
            x0_self_cond = None
            if i % 2 == 0:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    pred_eps = model(source_ids, source_mask, xt, t)
                    x0_self_cond = diffusion.predict_x0_from_eps(xt, t, pred_eps).detach()

            with torch.amp.autocast("cuda", dtype=torch.float16):
                predicted_eps = model(source_ids, source_mask, xt, t, x0_self_cond=x0_self_cond)

            mask = target_mask.unsqueeze(-1).float()
            loss = ((predicted_eps.float() - noise) ** 2 * mask).sum() / mask.sum() / config.embed_dim
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        torch.cuda.synchronize()
        peak = get_gpu_mem_mb()

        del optimizer, scaler, source_ids, source_mask, target_ids, target_mask
        reset_memory()
        return peak

    except torch.cuda.OutOfMemoryError:
        del optimizer, scaler
        reset_memory()
        model.zero_grad(set_to_none=True)
        return -1


def profile_inference(model, config, diffusion, emb_scale, device, batch_size=1, embed_dim=None):
    """Simulate inference: full reverse diffusion loop (a few steps) on single GPU."""
    reset_memory()
    model.eval()
    embed_dim = embed_dim or config.embed_dim

    source_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    source_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)

    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            B = source_ids.shape[0]
            xt = torch.randn(B, config.max_seq_len, embed_dim, device=device)
            x0_self_cond = None

            # Simulate a few reverse steps (memory is the same for all steps)
            for t_int in [199, 198, 197, 196, 195]:
                t = torch.full((B,), t_int, device=device, dtype=torch.long)
                predicted_eps = model(source_ids, source_mask, xt, t, x0_self_cond=x0_self_cond)
                x0_pred = diffusion.predict_x0_from_eps(xt, t, predicted_eps)
                x0_self_cond = x0_pred.detach()

                if t_int > 0:
                    alpha_bar_t = diffusion.alpha_bar[t_int]
                    alpha_bar_prev = diffusion.alpha_bar[t_int - 1]
                    beta_t = diffusion.betas[t_int]
                    coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
                    coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(diffusion.alphas[t_int]) / (1.0 - alpha_bar_t)
                    mean = coef_x0 * x0_pred + coef_xt * xt
                    variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    xt = mean + torch.sqrt(variance) * torch.randn_like(xt)

        torch.cuda.synchronize()
        peak = get_gpu_mem_mb()

        del source_ids, source_mask, xt
        reset_memory()
        return peak

    except torch.cuda.OutOfMemoryError:
        reset_memory()
        return -1


# Model configurations to test: (embed_dim, num_heads, num_layers, ff_dim, label)
MODEL_CONFIGS = [
    # Current
    (768,  12, 12, 3072,  "current (BERT-base)"),
    # Wider
    (1024, 16, 12, 4096,  "1024d 12L (wider)"),
    # Deeper
    (768,  12, 18, 3072,  "768d 18L (deeper)"),
    # Wider + deeper
    (1024, 16, 16, 4096,  "1024d 16L"),
    # BERT-large equivalent
    (1024, 16, 24, 4096,  "BERT-large (1024d 24L)"),
    # In between
    (896,  14, 16, 3584,  "896d 16L (mid)"),
    (896,  14, 20, 3584,  "896d 20L"),
    (1024, 16, 20, 4096,  "1024d 20L"),
    # Bigger
    (1280, 20, 12, 5120,  "1280d 12L"),
    (1280, 20, 16, 5120,  "1280d 16L"),
]


def main():
    device = torch.device("cuda:0")
    config = Config()

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {gpu_name} ({gpu_mem_total:.0f} MB)")
    print(f"Constraint: inference must fit on 1x GPU, training on 2x GPUs w/ checkpointing")
    print()

    # Training batch sizes to test (per GPU)
    train_batch = 64  # conservative starting point for larger models
    inference_batches = [1, 8, 16]  # typical inference batch sizes

    print(f"{'Config':<28} | {'Params':>8} | {'Train(bs={})'.format(train_batch):>12} | {'Inf bs=1':>9} | {'Inf bs=8':>9} | {'Inf bs=16':>9} | Verdict")
    print(f"{'-'*28}-+-{'-'*8}-+-{'-'*12}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+--------")

    for embed_dim, num_heads, num_layers, ff_dim, label in MODEL_CONFIGS:
        reset_memory()

        model = DiffusionTransformer(
            vocab_size=120000,  # approx bert-base-multilingual-cased
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=0.1,
            max_seq_len=config.max_seq_len,
        ).to(device)

        params_m = count_params(model) / 1e6

        with torch.no_grad():
            emb_scale = model.token_embedding.weight.norm(dim=-1).mean().item()

        diffusion = GaussianDiffusion(
            timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(device)

        # Inference profiling (no checkpointing needed)
        inf_results = {}
        for inf_bs in inference_batches:
            inf_mem = profile_inference(model, config, diffusion, emb_scale, device, batch_size=inf_bs, embed_dim=embed_dim)
            inf_results[inf_bs] = inf_mem

        # Training profiling (with gradient checkpointing)
        model_ckpt = enable_gradient_checkpointing(model)

        # Update config temporarily
        config_copy = Config()
        config_copy.embed_dim = embed_dim
        config_copy.num_heads = num_heads
        config_copy.num_layers = num_layers
        config_copy.ff_dim = ff_dim

        train_mem = profile_training(model_ckpt, config_copy, diffusion, emb_scale, device, train_batch)

        # Verdict
        train_ok = train_mem > 0 and train_mem < gpu_mem_total * 0.90
        inf_ok = inf_results[1] > 0 and inf_results[1] < gpu_mem_total * 0.90
        if train_ok and inf_ok:
            verdict = "OK"
        elif not inf_ok:
            verdict = "INF OOM"
        elif not train_ok:
            verdict = "TRAIN OOM"
        else:
            verdict = "BOTH OOM"

        def fmt_mem(m):
            return f"{m:.0f} MB" if m > 0 else "OOM"

        print(f"{label:<28} | {params_m:>6.0f}M | {fmt_mem(train_mem):>12} | {fmt_mem(inf_results[1]):>9} | {fmt_mem(inf_results[8]):>9} | {fmt_mem(inf_results[16]):>9} | {verdict}")

        del model, model_ckpt, diffusion
        reset_memory()

    print()
    print("Train = training with grad checkpointing, bs=64/GPU, AMP fp16")
    print("Inf = inference with no_grad + AMP fp16, 5 diffusion steps")
    print("Target: both under 90% of 24GB = 21,623 MB")


if __name__ == "__main__":
    main()
