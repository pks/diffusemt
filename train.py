import os
import sys
import math
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import Config
from model import DiffusionTransformer
from diffusion import MaskDiffusion
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
def validate(model, diffusion, val_dataloader, config, device):
    """Run validation and return average cross-entropy loss on masked target positions."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_dataloader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_mask = batch["target_mask"].to(device)

        B = target_ids.shape[0]
        S = source_ids.shape[1]
        t = torch.randint(1, config.timesteps + 1, (B,), device=device)

        # Corrupt target
        corrupted_target, is_masked = diffusion.q_sample(target_ids, t)

        # Build concatenated input
        input_ids, padding_mask, segment_ids, _ = diffusion._build_input(
            source_ids, source_mask, corrupted_target, target_mask)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids, padding_mask, segment_ids, t)

        # Loss on masked target positions only
        target_logits = logits[:, S:]
        loss_mask = is_masked & target_mask
        if loss_mask.sum() > 0:
            loss = torch.nn.functional.cross_entropy(
                target_logits[loss_mask], target_ids[loss_mask])
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


def enable_gradient_checkpointing(model):
    """Wrap each encoder layer with activation checkpointing."""
    from torch.utils.checkpoint import checkpoint

    for i, layer in enumerate(model.encoder.layers):
        original_forward = layer.forward

        def make_ckpt_forward(orig_fn):
            def ckpt_forward(src, src_mask=None, src_key_padding_mask=None, is_causal=None):
                def fn(s, m, kpm):
                    return orig_fn(s, src_mask=m, src_key_padding_mask=kpm)
                return checkpoint(fn, src, src_mask, src_key_padding_mask, use_reentrant=False)
            return ckpt_forward

        layer.forward = make_ckpt_forward(original_forward)


@torch.no_grad()
def health_check(model, diffusion, config, device,
                 dataloader_iter, dataloader, sampler, epoch):
    """Check for mode collapse and measure per-timestep accuracy."""
    model.eval()

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

    B = target_ids.shape[0]
    S = source_ids.shape[1]
    real_mask = target_mask.bool()

    results = {}
    for t_val in [1, config.timesteps // 4, config.timesteps // 2,
                  3 * config.timesteps // 4, config.timesteps]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        corrupted_target, is_masked = diffusion.q_sample(target_ids, t)

        input_ids, padding_mask, segment_ids, _ = diffusion._build_input(
            source_ids, source_mask, corrupted_target, target_mask)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(input_ids, padding_mask, segment_ids, t)

        target_logits = logits[:, S:]
        pred_tokens = target_logits.argmax(dim=-1)

        # Accuracy on masked real target positions
        masked_real = is_masked & real_mask
        if masked_real.sum() > 0:
            acc = (pred_tokens[masked_real] == target_ids[masked_real]).float().mean().item()
            n_unique = pred_tokens[masked_real].unique().numel()
        else:
            acc = 1.0
            n_unique = -1

        results[f"t{t_val}_acc"] = acc
        results[f"t{t_val}_unique"] = n_unique

    model.train()

    # Collapse detection
    t_low = 1
    t_low_unique = results.get(f"t{t_low}_unique", 0)
    collapsed = 0 < t_low_unique < 5

    return results, collapsed


def train(resume_from=None):
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

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    vocab_size = tokenizer.vocab_size

    # max_seq_len for model = 2 * per-side max (source + target concatenated)
    model = DiffusionTransformer(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len * 2,
    ).to(device)

    enable_gradient_checkpointing(model)

    if distributed:
        model = DDP(model, device_ids=[local_rank])

    raw_model = model.module if distributed else model

    if is_main:
        n_params = sum(p.numel() for p in raw_model.parameters())
        n_tied = raw_model.output_proj.weight.numel()
        print(f"Model params: {(n_params - n_tied) / 1e6:.1f}M (+ {n_tied / 1e6:.1f}M tied)")

    diffusion = MaskDiffusion(
        timesteps=config.timesteps,
        mask_token_id=config.mask_token_id,
        schedule=config.schedule,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    warmup_steps = getattr(config, 'warmup_steps', 0)
    # Total optimizer steps for cosine decay
    total_opt_steps = config.num_train_steps // config.grad_accum_steps
    def lr_lambda(current_step):
        # Linear warmup
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
        # Cosine decay to 0 after warmup
        progress = (current_step - warmup_steps) / max(total_opt_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler("cuda")

    dataloader, sampler = get_dataloader(config, split="train", distributed=distributed)
    val_dataloader, _ = get_dataloader(config, split="test", distributed=False)

    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Resume
    step = 0
    epoch = 0
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        step = ckpt["step"]
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        for _ in range(step // config.grad_accum_steps):
            scheduler.step()
        del ckpt
        if is_main:
            print(f"Resumed from checkpoint at step {step}")

    health_check_steps = {500, 1000, 2000, 5000}
    log_path = os.path.join(config.checkpoint_dir, "metrics.jsonl")

    def log_metrics(metrics):
        if is_main:
            with open(log_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

    loss_window = []
    PLATEAU_WINDOW = 20
    PLATEAU_THRESHOLD = 0.005

    model.train()
    running_loss = 0.0

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

            B = target_ids.shape[0]
            S = source_ids.shape[1]

            # Sample timesteps (1..T)
            t = torch.randint(1, config.timesteps + 1, (B,), device=device)

            # Forward diffusion: mask target tokens
            corrupted_target, is_masked = diffusion.q_sample(target_ids, t)

            # Build [source | corrupted_target] input
            input_ids, padding_mask, segment_ids, _ = diffusion._build_input(
                source_ids, source_mask, corrupted_target, target_mask)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(input_ids, padding_mask, segment_ids, t)

            # Cross-entropy on masked target positions only (fp32 for 120K vocab)
            target_logits = logits[:, S:].float()  # (B, T, V)
            loss_mask = is_masked & target_mask
            if loss_mask.sum() > 0:
                loss = torch.nn.functional.cross_entropy(
                    target_logits[loss_mask], target_ids[loss_mask],
                    label_smoothing=config.label_smoothing)
            else:
                loss = torch.tensor(0.0, device=device)

            loss = loss / config.grad_accum_steps
            scaler.scale(loss).backward()

            running_loss += loss.item() * config.grad_accum_steps

            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            step += 1

            if is_main and step % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                lr_now = scheduler.get_last_lr()[0]
                print(f"Step {step}/{config.num_train_steps} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")
                log_metrics({"step": step, "train_loss": round(avg_loss, 4), "lr": lr_now})
                running_loss = 0.0

                loss_window.append(avg_loss)
                if len(loss_window) > PLATEAU_WINDOW:
                    loss_window.pop(0)
                if len(loss_window) == PLATEAU_WINDOW and step > config.warmup_steps * config.grad_accum_steps:
                    spread = max(loss_window) - min(loss_window)
                    if spread < PLATEAU_THRESHOLD:
                        print(f"WARNING: Loss plateau detected (spread={spread:.4f} over {PLATEAU_WINDOW} intervals)")
                        log_metrics({"step": step, "event": "plateau_warning", "spread": round(spread, 4)})

            if is_main and (step in health_check_steps or step % config.val_every == 0):
                hc_results, collapsed = health_check(
                    model, diffusion, config, device,
                    iter(dataloader), dataloader, sampler, epoch)
                t_low = 1
                t_high = config.timesteps
                print(f"Step {step} | Health: t={t_low} acc={hc_results.get(f't{t_low}_acc', 0):.2%} "
                      f"unique={hc_results.get(f't{t_low}_unique', 0)}, "
                      f"t={t_high} acc={hc_results.get(f't{t_high}_acc', 0):.2%} "
                      f"unique={hc_results.get(f't{t_high}_unique', 0)}")
                log_metrics({"step": step, "event": "health_check", **hc_results})

                if collapsed:
                    print(f"FATAL: Mode collapse detected at step {step}! "
                          f"t={t_low} unique tokens = {hc_results.get(f't{t_low}_unique', 0)}. Aborting.")
                    log_metrics({"step": step, "event": "collapse_abort"})
                    if distributed:
                        cleanup_ddp()
                    sys.exit(1)

            if is_main and step % config.val_every == 0:
                val_loss = validate(model, diffusion, val_dataloader, config, device)
                print(f"Step {step} | Val Loss: {val_loss:.4f}")
                log_metrics({"step": step, "val_loss": round(val_loss, 4)})

            if is_main and step % config.save_every == 0:
                path = os.path.join(config.checkpoint_dir, f"model_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": config,
                }, path)
                print(f"Saved checkpoint: {path}")

    if is_main:
        path = os.path.join(config.checkpoint_dir, "model_final.pt")
        torch.save({
            "step": step,
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
        }, path)
        print(f"Training complete. Final checkpoint: {path}")

    if distributed:
        cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(resume_from=args.resume)
