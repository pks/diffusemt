import os
import sys
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import Config
from model import DiffusionTransformer
from diffusion import GaussianDiffusion
from dataset import get_dataloader
from transformers import AutoTokenizer


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


@torch.no_grad()
def validate(model, raw_model, diffusion, val_dataloader, config, device, emb_scale):
    """Run validation and return average x0-prediction MSE loss (min-SNR weighted)."""
    raw_model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in val_dataloader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_mask = batch["target_mask"].to(device)

        x0 = raw_model.token_embedding(target_ids) / emb_scale
        B = x0.shape[0]
        t = torch.randint(0, config.timesteps, (B,), device=device)
        xt, noise = diffusion.q_sample(x0, t)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_x0 = raw_model(source_ids, source_mask, xt, t).detach()
            predicted_x0 = raw_model(source_ids, source_mask, xt, t,
                                 x0_self_cond=pred_x0)

        mask = target_mask.unsqueeze(-1).float()
        per_sample_mse = ((predicted_x0.float() - x0) ** 2 * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)) / config.embed_dim

        # Min-SNR weighting
        snr = diffusion.snr[t]
        weight = torch.clamp(snr, max=config.min_snr_gamma) / snr
        loss = (weight * per_sample_mse).mean()

        total_loss += loss.item()
        total_batches += 1

    raw_model.train()
    return total_loss / total_batches


def enable_gradient_checkpointing(model):
    """Wrap each encoder/decoder layer with activation checkpointing."""
    from torch.utils.checkpoint import checkpoint

    for i, layer in enumerate(model.source_encoder.layers):
        original_forward = layer.forward

        def make_ckpt_forward(orig_fn):
            def ckpt_forward(src, src_mask=None, src_key_padding_mask=None, is_causal=None):
                def fn(s, m, kpm):
                    return orig_fn(s, src_mask=m, src_key_padding_mask=kpm)
                return checkpoint(fn, src, src_mask, src_key_padding_mask, use_reentrant=False)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_forward)

    for i, layer in enumerate(model.target_decoder.layers):
        original_forward = layer.forward

        def make_ckpt_forward(orig_fn):
            def ckpt_forward(tgt, memory, tgt_mask=None, memory_mask=None,
                             tgt_key_padding_mask=None, memory_key_padding_mask=None,
                             tgt_is_causal=None, memory_is_causal=None):
                def fn(t, m, mkpm):
                    return orig_fn(t, m, memory_key_padding_mask=mkpm)
                return checkpoint(fn, tgt, memory, memory_key_padding_mask, use_reentrant=False)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_forward)


@torch.no_grad()
def health_check(model, raw_model, diffusion, config, device, emb_scale,
                 dataloader_iter, dataloader, sampler, epoch):
    """Check for mode collapse and measure per-timestep accuracy.

    Returns a dict with diagnostics. Aborts training if collapse detected.
    """
    raw_model.eval()

    # Get a batch
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        if sampler is not None:
            sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

    source_ids = batch["source_ids"].to(device)
    source_mask = batch["source_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    target_mask = batch["target_mask"].to(device)

    # Use a small subset to avoid OOM on (B, T, V) distance matrix
    HC_BATCH = 16
    source_ids = source_ids[:HC_BATCH]
    source_mask = source_mask[:HC_BATCH]
    target_ids = target_ids[:HC_BATCH]
    target_mask = target_mask[:HC_BATCH]

    x0 = raw_model.token_embedding(target_ids) / emb_scale
    emb_weight = raw_model.token_embedding.weight.data
    B = x0.shape[0]

    results = {}
    for t_val in [0, config.timesteps // 4, config.timesteps // 2,
                  3 * config.timesteps // 4, config.timesteps - 1]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        xt, _ = diffusion.q_sample(x0, t)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred_x0 = raw_model(source_ids, source_mask, xt, t)

        # Token matching on GPU
        pred_scaled = (pred_x0.float() * emb_scale)
        token_ids = torch.cdist(pred_scaled, emb_weight.unsqueeze(0).float(), p=2).argmin(dim=-1)
        real_mask = target_mask.bool()
        acc = (token_ids[real_mask] == target_ids[real_mask]).float().mean().item()

        # Check unique tokens predicted (collapse = 1-2 unique tokens)
        n_unique = token_ids[real_mask].unique().numel()

        results[f"t{t_val}_acc"] = acc
        results[f"t{t_val}_unique"] = n_unique

    raw_model.train()

    # Collapse detection: if t=0 (clean input) has <5 unique tokens, it's collapsed
    t0_unique = results.get(f"t0_unique", 0)
    collapsed = t0_unique < 5

    return results, collapsed


def train(resume_from=None, fresh_optim=False):
    config = Config()

    # DDP setup
    distributed = "RANK" in os.environ
    if distributed:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    if is_main:
        print(f"Using {world_size} GPU(s), grad_accum_steps={config.grad_accum_steps}")
        effective_batch = config.batch_size * world_size * config.grad_accum_steps
        print(f"Effective batch size: {effective_batch}")

    # Tokenizer (for vocab size)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = tokenizer.vocab_size

    # Model
    model = DiffusionTransformer(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
    ).to(device)

    enable_gradient_checkpointing(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if distributed else model

    # Diffusion
    diffusion = GaussianDiffusion(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule=config.schedule,
    ).to(device)

    # Optimizer + lr schedule: linear warmup then cosine decay to min_lr
    import math
    import copy
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    warmup_steps = getattr(config, 'warmup_steps', 0)
    total_steps = config.num_train_steps
    min_lr_ratio = 0.1  # decay to 10% of peak LR, not zero
    def lr_lambda(current_step):
        if warmup_steps > 0 and current_step < warmup_steps:
            return current_step / warmup_steps
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # EMA: exponential moving average of model weights for stable inference
    # Keep on CPU to avoid GPU OOM. Update every 10 steps to minimize GPU→CPU copy overhead.
    ema_decay = 0.9999
    ema_update_every = 10
    ema_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
    # Build name→param mapping for fast iteration (avoid state_dict() overhead)
    def _ema_param_iter():
        for name, param in raw_model.named_parameters():
            yield name, param.data
        for name, buf in raw_model.named_buffers():
            yield name, buf
    def update_ema():
        with torch.no_grad():
            # Effective decay for N steps: decay^N
            effective_decay = ema_decay ** ema_update_every
            for key, param in _ema_param_iter():
                ema_state[key].mul_(effective_decay).add_(param.cpu(), alpha=1 - effective_decay)

    # Mixed precision (bfloat16 — no scaler needed)

    # Data
    dataloader, sampler = get_dataloader(config, split="train", distributed=distributed)
    val_dataloader, _ = get_dataloader(config, split="test", distributed=False)

    # Checkpoint dir
    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Compute embedding scale factor (normalize to unit per-dim variance for diffusion)
    with torch.no_grad():
        emb_scale = raw_model.token_embedding.weight.std().item()

    # Resume from checkpoint
    step = 0
    epoch = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        step = ckpt["step"]
        emb_scale = ckpt.get("emb_scale", emb_scale)
        if "ema_model" in ckpt:
            ema_state = {k: v.cpu() for k, v in ckpt["ema_model"].items()}
        else:
            ema_state = {k: v.cpu().clone() for k, v in raw_model.state_dict().items()}
        if not fresh_optim:
            optimizer.load_state_dict(ckpt["optimizer"])
            # Move optimizer state to device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        # Advance scheduler to match resumed step
        for _ in range(step // config.grad_accum_steps):
            scheduler.step()
        del ckpt
        if is_main:
            print(f"Resumed from checkpoint at step {step}"
                  f"{' (fresh optimizer)' if fresh_optim else ''}")

    if is_main:
        print(f"Embedding scale factor: {emb_scale:.2f}")

    # Health check interval: first few checks early, then at val_every
    health_check_steps = {500, 1000, 2000, 5000}

    # Log file for structured metrics
    log_path = os.path.join(config.checkpoint_dir, "metrics.jsonl")

    def log_metrics(metrics):
        if is_main:
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    # Loss plateau detection
    loss_window = []
    PLATEAU_WINDOW = 20  # number of log intervals
    PLATEAU_THRESHOLD = 0.005  # if max-min < threshold over window, plateau

    # Training loop
    model.train()
    running_loss = 0.0
    grad_norm = 0.0

    while step < config.num_train_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1

        for batch in dataloader:
            if step >= config.num_train_steps:
                break

            source_ids = batch["source_ids"].to(device)
            source_mask = batch["source_mask"].to(device)
            target_ids = batch["target_ids"].to(device)
            target_mask = batch["target_mask"].to(device)

            # Get clean target embeddings, normalized to unit scale
            with torch.no_grad():
                x0 = raw_model.token_embedding(target_ids) / emb_scale

            # Sample random timesteps
            B = x0.shape[0]
            t = torch.randint(0, config.timesteps, (B,), device=device)

            # Forward diffusion: add noise
            xt, noise = diffusion.q_sample(x0, t)

            # Per-sample source-init: 35% of samples use noisy source embeddings as xt
            # (target x0 stays the same — model must translate). Per-sample mixing
            # gives smooth gradients vs per-batch which creates conflicting updates.
            with torch.no_grad():
                x0_src = raw_model.token_embedding(source_ids) / emb_scale
                src_init_mask = torch.rand(B, device=device) < 0.35  # (B,)
                xt_src, _ = diffusion.q_sample(x0_src, t)
                xt = torch.where(src_init_mask[:, None, None], xt_src, xt)

            # Self-conditioning: 50% of the time, get a first prediction
            # and use it as conditioning for the real prediction
            x0_self_cond = None
            if torch.rand(1).item() > 0.5:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    x0_self_cond = model(source_ids, source_mask, xt, t).detach()

            # Model predicts x0 (clean embeddings) — mixed precision forward
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predicted_x0 = model(source_ids, source_mask, xt, t,
                                     x0_self_cond=x0_self_cond)

            # MSE loss on x0 prediction, per-sample (fp32)
            mask = target_mask.unsqueeze(-1).float()  # (B, T, 1)
            per_sample_mse = ((predicted_x0.float() - x0) ** 2 * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)) / config.embed_dim

            # Min-SNR weighting: downweight high-noise timesteps
            snr = diffusion.snr[t]
            weight = torch.clamp(snr, max=config.min_snr_gamma) / snr  # (B,)
            mse_loss = (weight * per_sample_mse).mean()

            # Auxiliary CE loss: force predicted embeddings toward correct discrete tokens
            # Use a small subset (64 samples) to avoid OOM on (B, T, V) logits
            ce_weight = getattr(config, 'ce_loss_weight', 0.0)
            if ce_weight > 0:
                CE_BATCH = 64
                emb_w = raw_model.token_embedding.weight.data  # (V, D)
                pred_sub = predicted_x0[:CE_BATCH].float() * emb_scale
                tgt_sub = target_ids[:CE_BATCH]
                mask_sub = target_mask[:CE_BATCH].bool()
                logits = 2 * torch.matmul(pred_sub, emb_w.T) - (emb_w ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)
                flat_logits = logits[mask_sub]  # (N, V)
                flat_targets = tgt_sub[mask_sub]  # (N,)
                ce_loss = torch.nn.functional.cross_entropy(flat_logits, flat_targets)
                ce_loss = torch.clamp(ce_loss, max=5.0)  # prevent gradient explosion
                loss = (mse_loss + ce_weight * ce_loss) / config.grad_accum_steps
            else:
                loss = mse_loss / config.grad_accum_steps
            loss.backward()

            running_loss += loss.item() * config.grad_accum_steps

            # Step optimizer every grad_accum_steps
            if (step + 1) % config.grad_accum_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                if step % ema_update_every == 0:
                    update_ema()

            step += 1

            if is_main and step % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"Step {step}/{config.num_train_steps} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e} | GradNorm: {grad_norm:.2f}")
                log_metrics({"step": step, "train_loss": round(avg_loss, 4), "lr": lr_now, "grad_norm": round(grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm, 4)})
                running_loss = 0.0

                # Plateau detection
                loss_window.append(avg_loss)
                if len(loss_window) > PLATEAU_WINDOW:
                    loss_window.pop(0)
                if len(loss_window) == PLATEAU_WINDOW and step > config.warmup_steps * config.grad_accum_steps:
                    spread = max(loss_window) - min(loss_window)
                    if spread < PLATEAU_THRESHOLD:
                        print(f"WARNING: Loss plateau detected (spread={spread:.4f} over {PLATEAU_WINDOW} intervals)")
                        log_metrics({"step": step, "event": "plateau_warning", "spread": round(spread, 4)})

            # Health check: early steps + at validation intervals (all ranks participate)
            if step in health_check_steps or step % config.val_every == 0:
                hc_results, collapsed = health_check(
                    model, raw_model, diffusion, config, device, emb_scale,
                    iter(dataloader), dataloader, sampler, epoch)
                if is_main:
                    t0_key = f"t0_acc"
                    tmax_key = f"t{config.timesteps - 1}_acc"
                    print(f"Step {step} | Health: t=0 acc={hc_results.get(t0_key, 0):.2%} "
                          f"unique={hc_results.get('t0_unique', 0)}, "
                          f"t={config.timesteps-1} acc={hc_results.get(tmax_key, 0):.2%} "
                          f"unique={hc_results.get(f't{config.timesteps-1}_unique', 0)}")
                    log_metrics({"step": step, "event": "health_check", **hc_results})

                if collapsed:
                    if is_main:
                        print(f"FATAL: Mode collapse detected at step {step}! "
                              f"t=0 unique tokens = {hc_results.get('t0_unique', 0)}. Aborting.")
                        log_metrics({"step": step, "event": "collapse_abort"})
                    if distributed:
                        cleanup_ddp()
                    sys.exit(1)

            if step % config.val_every == 0:
                val_loss = validate(model, raw_model, diffusion, val_dataloader,
                                    config, device, emb_scale)
                if is_main:
                    print(f"Step {step} | Val Loss: {val_loss:.4f}")
                    log_metrics({"step": step, "val_loss": round(val_loss, 4)})

            # Save more frequently early on (every 500 steps for first 5000)
            save_now = (step % config.save_every == 0) or (step <= 5000 and step % 500 == 0)
            if is_main and save_now:
                path = os.path.join(config.checkpoint_dir, f"model_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model": raw_model.state_dict(),
                    "ema_model": ema_state,
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                    "emb_scale": emb_scale,
                }, path)
                print(f"Saved checkpoint: {path}")

    # Save final checkpoint
    if is_main:
        path = os.path.join(config.checkpoint_dir, "model_final.pt")
        torch.save({
            "step": step,
            "model": raw_model.state_dict(),
            "ema_model": ema_state,
            "optimizer": optimizer.state_dict(),
            "config": config,
            "emb_scale": emb_scale,
        }, path)
        print(f"Training complete. Final checkpoint: {path}")

    if distributed:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--fresh-optim", action="store_true", help="Don't load optimizer state when resuming")
    args = parser.parse_args()
    train(resume_from=args.resume, fresh_optim=args.fresh_optim)
