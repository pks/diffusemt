"""
Find the max training batch size for BERT-large (1024d 24L) with gradient checkpointing.
Also test 1024d 22L as a fallback.
"""

import gc
import time
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


def enable_gradient_checkpointing(model):
    original_encoder_forward = model.source_encoder.forward
    original_decoder_forward = model.target_decoder.forward

    def checkpointed_encoder_forward(src, mask=None, src_key_padding_mask=None, is_causal=None):
        def custom_forward(src_, mask_, src_key_padding_mask_):
            return original_encoder_forward(src_, mask=mask_, src_key_padding_mask=src_key_padding_mask_)
        return torch.utils.checkpoint.checkpoint(
            custom_forward, src, mask, src_key_padding_mask, use_reentrant=False)

    def checkpointed_decoder_forward(tgt, memory, tgt_mask=None, memory_mask=None,
                                      tgt_key_padding_mask=None, memory_key_padding_mask=None,
                                      tgt_is_causal=None, memory_is_causal=None):
        def custom_forward(tgt_, memory_, memory_key_padding_mask_):
            return original_decoder_forward(tgt_, memory_, memory_key_padding_mask=memory_key_padding_mask_)
        return torch.utils.checkpoint.checkpoint(
            custom_forward, tgt, memory, memory_key_padding_mask, use_reentrant=False)

    model.source_encoder.forward = checkpointed_encoder_forward
    model.target_decoder.forward = checkpointed_decoder_forward
    return model


def profile_training(model, embed_dim, config, diffusion, emb_scale, device, batch_size):
    reset_memory()
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
        del optimizer, scaler, source_ids, source_mask, target_ids, target_mask
        reset_memory()
        return peak
    except torch.cuda.OutOfMemoryError:
        del optimizer, scaler
        reset_memory()
        model.zero_grad(set_to_none=True)
        return -1


def main():
    device = torch.device("cuda:0")
    config = Config()
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_total:.0f} MB)")
    print()

    configs = [
        (1024, 16, 22, 4096, "1024d 22L"),
        (1024, 16, 24, 4096, "BERT-large (1024d 24L)"),
    ]

    batch_sizes = [8, 16, 24, 32, 40, 48, 56, 64]

    for embed_dim, num_heads, num_layers, ff_dim, label in configs:
        print(f"{'='*65}")
        print(f"  {label}  (embed={embed_dim}, layers={num_layers}, ff={ff_dim})")

        model = DiffusionTransformer(
            vocab_size=120000,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim,
            dropout=0.1,
            max_seq_len=config.max_seq_len,
        ).to(device)

        params_m = count_params(model) / 1e6
        print(f"  Parameters: {params_m:.0f}M")

        with torch.no_grad():
            emb_scale = model.token_embedding.weight.norm(dim=-1).mean().item()

        diffusion = GaussianDiffusion(
            timesteps=config.timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(device)

        model = enable_gradient_checkpointing(model)

        print(f"{'='*65}")
        print(f"  {'Batch':>5} | {'Peak MB':>9} | {'% Used':>7} | {'Eff batch (2GPU×accum)':>22} | Status")
        print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*7}-+-{'-'*22}-+-------")

        max_bs = 0
        for bs in batch_sizes:
            peak = profile_training(model, embed_dim, config, diffusion, emb_scale, device, bs)
            if peak < 0:
                print(f"  {bs:>5} | {'OOM':>9} | {'---':>7} | {'---':>22} | OOM")
                break
            pct = peak / gpu_total * 100
            # Show effective batch for different grad_accum options
            eff_1 = bs * 2 * 1
            eff_2 = bs * 2 * 2
            eff_4 = bs * 2 * 4
            status = "OK" if pct < 92 else "TIGHT"
            print(f"  {bs:>5} | {peak:>8.0f} | {pct:>6.1f}% | {eff_1:>3} / {eff_2:>3} / {eff_4:>3} (a=1/2/4) | {status}")
            if pct < 95:
                max_bs = bs

        print()
        if max_bs > 0:
            print(f"  >> Max safe batch_size: {max_bs}")
            print(f"     With grad_accum=2:  effective batch = {max_bs * 2 * 2}")
            print(f"     With grad_accum=4:  effective batch = {max_bs * 2 * 4}")
            print(f"     With grad_accum=8:  effective batch = {max_bs * 2 * 8}")
        else:
            print(f"  >> Cannot fit even batch_size=8!")
        print()

        del model, diffusion
        reset_memory()


if __name__ == "__main__":
    main()
