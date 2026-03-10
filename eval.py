"""Evaluate a checkpoint: per-timestep accuracy, translation, and infilling."""
import argparse
import torch
from config import Config
from model import DiffusionTransformer
from diffusion import GaussianDiffusion
from translate import translate, infill, embeddings_to_tokens
from dataset import TranslationDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def per_timestep_accuracy(model, diffusion, config, device, emb_scale, n_samples=16):
    """Measure single-step denoising token accuracy at various timesteps."""
    emb_weight = model.token_embedding.weight.data

    ds = TranslationDataset("data/wmt14_en_de_tokenized")
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False)
    batch = next(iter(loader))

    source_ids = batch["source_ids"].to(device)
    source_mask = batch["source_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    target_mask = batch["target_mask"].to(device)

    x0 = model.token_embedding(target_ids) / emb_scale

    T = config.timesteps
    # Sample ~10 evenly spaced timesteps including 0 and T-1
    if T <= 10:
        test_timesteps = list(range(T))
    else:
        step = max(1, T // 10)
        test_timesteps = list(range(0, T, step))
        if T - 1 not in test_timesteps:
            test_timesteps.append(T - 1)

    print(f"\n{'t':>5} | {'alpha_bar':>10} | {'MSE/dim':>8} | {'TokenAcc':>8}")
    print("-" * 45)

    for t_val in test_timesteps:
        B = x0.shape[0]
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        xt, _ = diffusion.q_sample(x0, t)

        with torch.no_grad():
            pred_x0 = model(source_ids, source_mask, xt, t)
            pred_x0_sc = model(source_ids, source_mask, xt, t, x0_self_cond=pred_x0)

        mask = target_mask.unsqueeze(-1).float()
        mse = ((pred_x0_sc - x0) ** 2 * mask).sum() / mask.sum() / config.embed_dim

        pred_scaled = pred_x0_sc * emb_scale
        token_ids = torch.cdist(pred_scaled.float(), emb_weight.unsqueeze(0).float(), p=2).argmin(dim=-1)
        real_mask = target_mask.bool()
        correct = (token_ids[real_mask] == target_ids[real_mask]).float().mean().item()

        ab = diffusion.alpha_bar[t_val].item()
        print(f"{t_val:5d} | {ab:10.6f} | {mse:.4f} | {correct:7.2%}")


def eval_translations(model, diffusion, tokenizer, config, device, emb_scale):
    """Translate a few test sentences."""
    sentences = [
        "The weather is nice today.",
        "I have a cat.",
        "The European Parliament has approved the proposal.",
        "She went to the store to buy some milk.",
        "Hello, how are you?",
    ]

    print("\n=== Translations ===")
    for s in sentences:
        out = translate(s, model, diffusion, tokenizer, config, device, emb_scale)
        print(f"SRC: {s}")
        print(f"OUT: {out[:120]}")
        print()


def eval_infilling(model, diffusion, tokenizer, config, device, emb_scale):
    """Test infilling with known prefix/suffix."""
    tests = [
        ("The weather is nice today.", "Das Wetter ___ heute schön."),
        ("I have a cat.", "Ich habe ___ Katze."),
        ("The European Parliament has approved the proposal.",
         "Das Europäische Parlament hat ___ Vorschlag gebilligt."),
        ("She went to the store to buy some milk.",
         "Sie ging ___ um Milch zu kaufen."),
        # prefix only
        ("The weather is nice today.", "Das Wetter ist ___"),
        # suffix only
        ("I have a cat.", "___ eine Katze."),
    ]

    print("\n=== Infilling ===")
    for src, partial in tests:
        out = infill(src, partial, model, diffusion, tokenizer, config, device, emb_scale)
        print(f"SRC: {src}")
        print(f"PARTIAL: {partial}")
        print(f"OUTPUT:  {out[:120]}")
        print()


def eval_infilling_accuracy(model, diffusion, config, device, emb_scale, n_samples=16):
    """Quantitative infilling: mask middle 20% of target, measure token accuracy."""
    emb_weight = model.token_embedding.weight.data
    norm_emb_weight = emb_weight / emb_scale

    ds = TranslationDataset("data/wmt14_en_de_tokenized")
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False)
    batch = next(iter(loader))

    source_ids = batch["source_ids"].to(device)
    source_mask = batch["source_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    target_mask = batch["target_mask"].to(device)

    # Build infill mask: mask out middle 20% of real tokens
    B, T = target_ids.shape
    infill_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for i in range(B):
        real_len = target_mask[i].sum().item()
        start = int(real_len * 0.4)
        end = int(real_len * 0.6)
        if end <= start:
            end = start + 1
        infill_mask[i, start:end] = True

    # Get known embeddings
    with torch.no_grad():
        known_x0 = model.token_embedding(target_ids) / emb_scale

    # Run infilling
    denoised = diffusion.p_sample_loop_infill(
        model, source_ids, source_mask,
        known_x0, infill_mask,
        seq_len=T, embed_dim=config.embed_dim,
        embedding_weight=norm_emb_weight,
    )

    denoised_scaled = denoised * emb_scale
    pred_ids = embeddings_to_tokens(denoised_scaled, emb_weight)

    # Accuracy on infilled positions only
    infill_correct = (pred_ids[infill_mask] == target_ids[infill_mask]).float().mean().item()
    # Accuracy on known positions (should be ~100%)
    known_mask = target_mask.bool() & ~infill_mask
    known_correct = (pred_ids[known_mask] == target_ids[known_mask]).float().mean().item()

    n_infilled = infill_mask.sum().item()
    print(f"\n=== Infilling accuracy (middle 20% masked, {n_infilled} positions) ===")
    print(f"Infilled positions: {infill_correct:.2%}")
    print(f"Known positions:    {known_correct:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--skip-translate", action="store_true", help="Skip slow translation eval")
    parser.add_argument("--skip-infill", action="store_true", help="Skip slow infilling eval")
    args = parser.parse_args()

    config = Config()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    model = DiffusionTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=0.0,
        max_seq_len=config.max_seq_len,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    emb_scale = checkpoint.get("emb_scale", 1.0)
    step = checkpoint["step"]
    del checkpoint
    print(f"Loaded checkpoint from step {step}, emb_scale={emb_scale:.4f}")

    diffusion = GaussianDiffusion(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
        schedule=config.schedule,
    ).to(device)

    model.eval()

    # Fast: per-timestep accuracy (just forward passes)
    per_timestep_accuracy(model, diffusion, config, device, emb_scale)

    # Slow: full reverse diffusion
    if not args.skip_translate:
        eval_translations(model, diffusion, tokenizer, config, device, emb_scale)

    if not args.skip_infill:
        eval_infilling(model, diffusion, tokenizer, config, device, emb_scale)
        eval_infilling_accuracy(model, diffusion, config, device, emb_scale)


if __name__ == "__main__":
    main()
