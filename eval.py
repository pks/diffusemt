"""Evaluate a source-corruption diffusion checkpoint."""
import argparse
import torch
from config import Config
from model import SourceCorruptionEncoderDecoder
from diffusion import SourceCorruptionDiffusion
from translate import translate, infill, build_model
from dataset import TranslationDataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def per_timestep_accuracy(model, diffusion, config, device, n_samples=16):
    """Measure single-step prediction accuracy at various corruption levels."""
    ds = TranslationDataset("data/wmt14_en_de_tokenized")
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False)
    batch = next(iter(loader))

    source_ids = batch["source_ids"].to(device)
    source_mask = batch["source_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    target_mask = batch["target_mask"].to(device)

    T = config.timesteps
    if T <= 10:
        test_timesteps = list(range(1, T + 1))
    else:
        step = max(1, T // 10)
        test_timesteps = list(range(1, T + 1, step))
        if T not in test_timesteps:
            test_timesteps.append(T)

    real_mask = target_mask.bool()

    print(f"\n{'t':>5} | {'gamma':>8} | {'corr%':>8} | {'CorrAcc':>8} | {'AllAcc':>8}")
    print("-" * 55)

    for t_val in test_timesteps:
        B = target_ids.shape[0]
        t = torch.full((B,), t_val, device=device, dtype=torch.long)
        corrupted, is_corrupted = diffusion.q_sample(
            source_ids, source_mask, target_ids, target_mask, t)

        with torch.no_grad():
            logits = model(corrupted, target_mask, t,
                           source_ids=source_ids, source_mask=source_mask)

        pred_tokens = logits.argmax(dim=-1)

        corrupted_real = is_corrupted & real_mask
        if corrupted_real.sum() > 0:
            corr_acc = (pred_tokens[corrupted_real] == target_ids[corrupted_real]).float().mean().item()
        else:
            corr_acc = 1.0

        all_acc = (pred_tokens[real_mask] == target_ids[real_mask]).float().mean().item()

        gamma = diffusion.gamma[t_val].item()
        pct_corrupted = is_corrupted[real_mask].float().mean().item() if real_mask.sum() > 0 else 0

        print(f"{t_val:5d} | {gamma:8.4f} | {pct_corrupted:7.1%} | {corr_acc:7.2%} | {all_acc:7.2%}")


def eval_translations(model, diffusion, tokenizer, config, device):
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
        out = translate(s, model, diffusion, tokenizer, config, device)
        print(f"SRC: {s}")
        print(f"OUT: {out[:120]}")
        print()


def eval_infilling(model, diffusion, tokenizer, config, device):
    """Test infilling with known prefix/suffix."""
    tests = [
        ("The weather is nice today.", "Das Wetter ___ heute schön."),
        ("I have a cat.", "Ich habe ___ Katze."),
        ("The European Parliament has approved the proposal.",
         "Das Europäische Parlament hat ___ Vorschlag gebilligt."),
        ("She went to the store to buy some milk.",
         "Sie ging ___ um Milch zu kaufen."),
        ("The weather is nice today.", "Das Wetter ist ___"),
        ("I have a cat.", "___ eine Katze."),
    ]

    print("\n=== Infilling ===")
    for src, partial in tests:
        out = infill(src, partial, model, diffusion, tokenizer, config, device)
        print(f"SRC: {src}")
        print(f"PARTIAL: {partial}")
        print(f"OUTPUT:  {out[:120]}")
        print()


def eval_infilling_accuracy(model, diffusion, config, device, n_samples=16):
    """Quantitative infilling: mask middle 20% of target, measure token accuracy."""
    ds = TranslationDataset("data/wmt14_en_de_tokenized")
    loader = DataLoader(ds, batch_size=n_samples, shuffle=False)
    batch = next(iter(loader))

    source_ids = batch["source_ids"].to(device)
    source_mask = batch["source_mask"].to(device)
    target_ids = batch["target_ids"].to(device)
    target_mask = batch["target_mask"].to(device)

    B, T = target_ids.shape
    infill_mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    for i in range(B):
        real_len = target_mask[i].sum().item()
        start = int(real_len * 0.4)
        end = int(real_len * 0.6)
        if end <= start:
            end = start + 1
        infill_mask[i, start:end] = True

    output_ids = diffusion.p_sample_loop_infill(
        model, source_ids, source_mask,
        target_ids, infill_mask, target_mask,
    )

    infill_correct = (output_ids[infill_mask] == target_ids[infill_mask]).float().mean().item()
    known_mask = target_mask.bool() & ~infill_mask
    known_correct = (output_ids[known_mask] == target_ids[known_mask]).float().mean().item()

    n_infilled = infill_mask.sum().item()
    print(f"\n=== Infilling accuracy (middle 20% masked, {n_infilled} positions) ===")
    print(f"Infilled positions: {infill_correct:.2%}")
    print(f"Known positions:    {known_correct:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--skip-translate", action="store_true")
    parser.add_argument("--skip-infill", action="store_true")
    args = parser.parse_args()

    config = Config()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    model = build_model(config, device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    step = checkpoint["step"]
    del checkpoint
    print(f"Loaded checkpoint from step {step}")

    diffusion = SourceCorruptionDiffusion(
        timesteps=config.timesteps,
        mask_token_id=config.mask_token_id,
        schedule=config.schedule,
    ).to(device)

    model.eval()
    per_timestep_accuracy(model, diffusion, config, device)

    if not args.skip_translate:
        eval_translations(model, diffusion, tokenizer, config, device)

    if not args.skip_infill:
        eval_infilling(model, diffusion, tokenizer, config, device)
        eval_infilling_accuracy(model, diffusion, config, device)


if __name__ == "__main__":
    main()
