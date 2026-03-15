import os
import sys
import math
import argparse
import json
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from config import Config
from model import SourceCorruptionEncoderDecoder, SourceCorruptionEncoderOnly
from diffusion import SourceCorruptionDiffusion
from dataset import get_dataloader


def setup_ddp():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp():
    dist.destroy_process_group()


def get_t_max(step, config):
    """Timestep curriculum for diffusion phase.

    Phase 1 (0..ar_steps): autoregressive — returns -1 (signal for AR mode)
    Phase 2a (ar_steps..curriculum_end_step): t_max ramps from t_start to T
    Phase 2b (curriculum_end_step+): t_max=T
    """
    T = config.timesteps
    if step < config.ar_steps:
        return -1  # autoregressive mode
    elif step < config.curriculum_end_step:
        progress = (step - config.ar_steps) / (
            config.curriculum_end_step - config.ar_steps)
        return max(int(config.curriculum_t_start + (T - config.curriculum_t_start) * progress), 1)
    else:
        return T


@torch.no_grad()
def validate(model, diffusion, val_dataloader, config, device):
    """Run validation and return average cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for batch in val_dataloader:
        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_ids = batch["target_ids"].to(device)
        target_mask = batch["target_mask"].to(device)

        B = target_ids.shape[0]
        t = torch.randint(1, config.timesteps + 1, (B,), device=device)

        corrupted, is_corrupted = diffusion.q_sample(
            source_ids, source_mask, target_ids, target_mask, t)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(corrupted, target_mask, t,
                           source_ids=source_ids, source_mask=source_mask)

        real_mask = target_mask.bool()
        if real_mask.sum() > 0:
            loss = torch.nn.functional.cross_entropy(
                logits[real_mask].float(), target_ids[real_mask])
            total_loss += loss.item()
            total_batches += 1

    model.train()
    return total_loss / max(total_batches, 1)


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
    real_mask = target_mask.bool()

    results = {}
    for t_val in [1, config.timesteps // 4, config.timesteps // 2,
                  3 * config.timesteps // 4, config.timesteps]:
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        corrupted, is_corrupted = diffusion.q_sample(
            source_ids, source_mask, target_ids, target_mask, t)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(corrupted, target_mask, t,
                           source_ids=source_ids, source_mask=source_mask)

        pred_tokens = logits.argmax(dim=-1)

        corrupted_real = is_corrupted & real_mask
        if corrupted_real.sum() > 0:
            acc = (pred_tokens[corrupted_real] == target_ids[corrupted_real]).float().mean().item()
            n_unique = pred_tokens[corrupted_real].unique().numel()
        else:
            acc = 1.0
            n_unique = -1

        results[f"t{t_val}_acc"] = acc
        results[f"t{t_val}_unique"] = n_unique

    model.train()

    t_mid = config.timesteps // 2
    t_mid_unique = results.get(f"t{t_mid}_unique", 0)
    collapsed = 0 < t_mid_unique < 5

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

    # Build model with frozen mBERT embeddings
    if config.architecture == "encoder-only":
        model = SourceCorruptionEncoderOnly(
            pretrained_name=config.pretrained_name,
            model_dim=config.model_dim,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            freeze_embeddings=config.freeze_embeddings,
        ).to(device)
    else:
        model = SourceCorruptionEncoderDecoder(
            pretrained_name=config.pretrained_name,
            model_dim=config.model_dim,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            encoder_layers=config.encoder_layers,
            decoder_layers=config.decoder_layers,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            max_seq_len=config.max_seq_len,
            freeze_embeddings=config.freeze_embeddings,
        ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = model.module if distributed else model

    if is_main:
        total_params = sum(p.numel() for p in raw_model.parameters())
        trainable_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        print(f"Model params: {total_params / 1e6:.1f}M total, "
              f"{trainable_params / 1e6:.1f}M trainable, "
              f"{frozen_params / 1e6:.1f}M frozen")

    diffusion = SourceCorruptionDiffusion(
        timesteps=config.timesteps,
        mask_token_id=config.mask_token_id,
        schedule=config.schedule,
    ).to(device)

    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config.lr, weight_decay=0.01)

    warmup_steps = config.warmup_steps
    total_opt_steps = config.num_train_steps // config.grad_accum_steps
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return current_step / max(warmup_steps, 1)
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

    model.train()
    running_loss = 0.0
    running_aux_loss = 0.0

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

            # Two-phase training
            t_max = get_t_max(step, config)

            if t_max == -1:
                # Phase 1: Autoregressive seq2seq with teacher forcing
                # Input: shifted target (prepend [CLS], drop last token)
                shifted = torch.zeros_like(target_ids)
                shifted[:, 0] = 101  # [CLS] token
                shifted[:, 1:] = target_ids[:, :-1]

                t = torch.ones(B, device=device, dtype=torch.long)  # dummy timestep

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(shifted, target_mask, t,
                                   source_ids=source_ids, source_mask=source_mask,
                                   causal=True)

                # Standard teacher-forcing CE loss
                loss_mask = target_mask.bool()
                if loss_mask.sum() > 0:
                    loss = torch.nn.functional.cross_entropy(
                        logits[loss_mask].float(), target_ids[loss_mask],
                        label_smoothing=config.label_smoothing)
                else:
                    loss = torch.tensor(0.0, device=device)
                is_corrupted = torch.zeros_like(target_mask)
            else:
                # Phase 2: Source-as-corruption diffusion
                t = torch.randint(1, t_max + 1, (B,), device=device)
                corrupted, is_corrupted = diffusion.q_sample(
                    source_ids, source_mask, target_ids, target_mask, t)

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    logits = model(corrupted, target_mask, t,
                                   source_ids=source_ids, source_mask=source_mask,
                                   causal=False)

                # x0-parameterization: CE on all real target positions
                loss_mask = target_mask.bool()
                if loss_mask.sum() > 0:
                    loss = torch.nn.functional.cross_entropy(
                        logits[loss_mask].float(), target_ids[loss_mask],
                        label_smoothing=config.label_smoothing)
                else:
                    loss = torch.tensor(0.0, device=device)

            # Auxiliary encoder MLM loss (only for encoder-decoder models)
            aux_loss = torch.tensor(0.0, device=device)
            m = model.module if distributed else model
            if config.aux_mlm_weight > 0 and hasattr(m, 'encode_source'):
                enc_out = m.encode_source(source_ids, source_mask)
                if enc_out is not None:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        aux_logits = m.aux_mlm_logits(enc_out)
                    # Mask 15% of source positions
                    src_real = source_mask.bool()
                    mlm_mask = torch.rand_like(source_mask.float()) < 0.15
                    mlm_mask = mlm_mask & src_real
                    if mlm_mask.sum() > 0:
                        aux_loss = torch.nn.functional.cross_entropy(
                            aux_logits[mlm_mask].float(), source_ids[mlm_mask])

            # Diversity regularization: encourage entropy in predictions (with gradients)
            div_loss = torch.tensor(0.0, device=device)
            if config.diversity_weight > 0 and loss_mask.sum() > 0:
                log_probs = torch.log_softmax(logits[loss_mask].float(), dim=-1)
                probs = log_probs.exp()
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                # Negative because we want to MAXIMIZE entropy (minimize negative entropy)
                div_loss = -entropy

            total_loss = loss + config.aux_mlm_weight * aux_loss + config.diversity_weight * div_loss
            total_loss = total_loss / config.grad_accum_steps
            scaler.scale(total_loss).backward()

            running_loss += loss.item()
            running_aux_loss += aux_loss.item()

            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            step += 1

            if is_main and step % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                avg_aux = running_aux_loss / config.log_every
                lr_now = scheduler.get_last_lr()[0]
                t_max_now = get_t_max(step, config)
                mode = "AR" if t_max_now == -1 else f"t_max={t_max_now}"
                print(f"Step {step}/{config.num_train_steps} | Loss: {avg_loss:.4f} | "
                      f"Aux: {avg_aux:.4f} | LR: {lr_now:.2e} | {mode}")
                log_metrics({"step": step, "train_loss": round(avg_loss, 4),
                             "aux_loss": round(avg_aux, 4), "lr": lr_now,
                             "t_max": t_max_now})
                running_loss = 0.0
                running_aux_loss = 0.0

            if is_main and (step in health_check_steps or step % config.val_every == 0) and get_t_max(step, config) != -1:
                hc_results, collapsed = health_check(
                    model, diffusion, config, device,
                    iter(dataloader), dataloader, sampler, epoch)
                t_mid = config.timesteps // 2
                t_high = config.timesteps
                print(f"Step {step} | Health: "
                      f"t={t_mid} acc={hc_results.get(f't{t_mid}_acc', 0):.2%} "
                      f"unique={hc_results.get(f't{t_mid}_unique', 0)}, "
                      f"t={t_high} acc={hc_results.get(f't{t_high}_acc', 0):.2%} "
                      f"unique={hc_results.get(f't{t_high}_unique', 0)}")
                log_metrics({"step": step, "event": "health_check", **hc_results})

                if collapsed:
                    print(f"FATAL: Mode collapse detected at step {step}! "
                          f"t={t_mid} unique tokens = {hc_results.get(f't{t_mid}_unique', 0)}. Aborting.")
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
