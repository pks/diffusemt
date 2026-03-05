import os
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
    """Run validation and return average epsilon-prediction MSE loss."""
    model.eval()
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

        with torch.amp.autocast("cuda", dtype=torch.float16):
            pred_eps = model(source_ids, source_mask, xt, t).detach()
            x0_self_cond = diffusion.predict_x0_from_eps(xt, t, pred_eps)
            predicted_eps = model(source_ids, source_mask, xt, t,
                                  x0_self_cond=x0_self_cond)

        mask = target_mask.unsqueeze(-1).float()
        loss = ((predicted_eps.float() - noise) ** 2 * mask).sum() / mask.sum() / config.embed_dim
        total_loss += loss.item()
        total_batches += 1

    model.train()
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


def train():
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
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda")

    # Data
    dataloader, sampler = get_dataloader(config, split="train", distributed=distributed)
    val_dataloader, _ = get_dataloader(config, split="test", distributed=False)

    # Checkpoint dir
    if is_main:
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    # Compute embedding scale factor (normalize to unit per-dim variance for diffusion)
    with torch.no_grad():
        emb_scale = raw_model.token_embedding.weight.std().item()
    if is_main:
        print(f"Embedding scale factor: {emb_scale:.2f}")

    # Training loop
    model.train()
    step = 0
    running_loss = 0.0
    epoch = 0

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

            # Self-conditioning: 50% of the time, get a first prediction
            # and use it as conditioning for the real prediction
            x0_self_cond = None
            if torch.rand(1).item() > 0.5:
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                    pred_eps = model(source_ids, source_mask, xt, t)
                    x0_self_cond = diffusion.predict_x0_from_eps(xt, t, pred_eps).detach()

            # Model predicts noise (epsilon) — mixed precision forward
            with torch.amp.autocast("cuda", dtype=torch.float16):
                predicted_eps = model(source_ids, source_mask, xt, t,
                                      x0_self_cond=x0_self_cond)

            # MSE loss on noise prediction, only on real token positions
            # Compute loss in fp32 for stability
            mask = target_mask.unsqueeze(-1).float()  # (B, T, 1)
            loss = ((predicted_eps.float() - noise) ** 2 * mask).sum() / mask.sum() / config.embed_dim
            loss = loss / config.grad_accum_steps
            scaler.scale(loss).backward()

            running_loss += loss.item() * config.grad_accum_steps

            # Step optimizer every grad_accum_steps
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            step += 1

            if is_main and step % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                print(f"Step {step}/{config.num_train_steps} | Loss: {avg_loss:.4f}")
                running_loss = 0.0

            if is_main and step % config.val_every == 0:
                val_loss = validate(model, raw_model, diffusion, val_dataloader,
                                    config, device, emb_scale)
                print(f"Step {step} | Val Loss: {val_loss:.4f}")

            if is_main and step % config.save_every == 0:
                path = os.path.join(config.checkpoint_dir, f"model_step_{step}.pt")
                torch.save({
                    "step": step,
                    "model": raw_model.state_dict(),
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
            "optimizer": optimizer.state_dict(),
            "config": config,
            "emb_scale": emb_scale,
        }, path)
        print(f"Training complete. Final checkpoint: {path}")

    if distributed:
        cleanup_ddp()


if __name__ == "__main__":
    train()
