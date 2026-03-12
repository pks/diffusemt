"""
Compare BERT-large training memory WITH vs WITHOUT gradient checkpointing.
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


def profile_step(model, embed_dim, config, diffusion, emb_scale, device, batch_size, warmup=2, measure=3):
    reset_memory()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    source_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    source_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)
    target_ids = torch.randint(0, 1000, (batch_size, config.max_seq_len), device=device)
    target_mask = torch.ones(batch_size, config.max_seq_len, dtype=torch.bool, device=device)

    model.train()
    step_times = []
    try:
        for i in range(warmup + measure):
            t0 = time.time()

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
            if i >= warmup:
                step_times.append(time.time() - t0)

        peak = get_gpu_mem_mb()
        avg_ms = sum(step_times) / len(step_times) * 1000
        samp_s = batch_size / (sum(step_times) / len(step_times))

        del optimizer, scaler, source_ids, source_mask, target_ids, target_mask
        reset_memory()
        return peak, avg_ms, samp_s

    except torch.cuda.OutOfMemoryError:
        del optimizer, scaler
        reset_memory()
        model.zero_grad(set_to_none=True)
        return -1, -1, -1


def main():
    device = torch.device("cuda:0")
    config = Config()
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
    print(f"GPU: {torch.cuda.get_device_name(0)} ({gpu_total:.0f} MB)")
    print()

    embed_dim, num_heads, num_layers, ff_dim = 1024, 16, 24, 4096
    batch_sizes = [8, 16, 24, 32, 40, 48, 56, 64]

    for use_ckpt in [False, True]:
        label = "WITH" if use_ckpt else "WITHOUT"
        print(f"{'='*70}")
        print(f"  BERT-large (1024d 24L, 834M) — {label} gradient checkpointing")
        print(f"{'='*70}")
        print(f"  {'Batch':>5} | {'Peak MB':>9} | {'% GPU':>7} | {'ms/step':>8} | {'samp/s':>7} | Status")
        print(f"  {'-'*5}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}-+-------")

        model = DiffusionTransformer(
            vocab_size=120000, embed_dim=embed_dim, num_heads=num_heads,
            num_layers=num_layers, ff_dim=ff_dim, dropout=0.1,
            max_seq_len=config.max_seq_len,
        ).to(device)

        with torch.no_grad():
            emb_scale = model.token_embedding.weight.norm(dim=-1).mean().item()

        diffusion = GaussianDiffusion(
            timesteps=config.timesteps, beta_start=config.beta_start,
            beta_end=config.beta_end,
        ).to(device)

        if use_ckpt:
            model = enable_gradient_checkpointing(model)

        for bs in batch_sizes:
            peak, avg_ms, samp_s = profile_step(
                model, embed_dim, config, diffusion, emb_scale, device, bs)

            if peak < 0:
                print(f"  {bs:>5} | {'OOM':>9} | {'---':>7} | {'---':>8} | {'---':>7} | OOM")
                break

            pct = peak / gpu_total * 100
            status = "OK" if pct < 90 else ("TIGHT" if pct < 95 else "DANGER")
            print(f"  {bs:>5} | {peak:>8.0f} | {pct:>6.1f}% | {avg_ms:>7.0f} | {samp_s:>6.0f} | {status}")

        print()
        del model, diffusion
        reset_memory()


if __name__ == "__main__":
    main()
