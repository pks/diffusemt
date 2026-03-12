import argparse
import torch
from config import Config
from model import DiffusionTransformer
from diffusion import MaskDiffusion
from transformers import AutoTokenizer


def translate(text, model, diffusion, tokenizer, config, device):
    """Translate a single source sentence."""
    src = tokenizer(
        text, max_length=config.max_seq_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    source_ids = src["input_ids"].to(device)
    source_mask = src["attention_mask"].to(device).bool()

    model.eval()
    output_ids = diffusion.p_sample_loop(
        model, source_ids, source_mask,
        target_len=config.max_seq_len,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def infill(text, partial, model, diffusion, tokenizer, config, device):
    """Fill in masked spans of a partial translation.

    partial: target text with ___ for spans to generate.
    """
    src = tokenizer(
        text, max_length=config.max_seq_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    source_ids = src["input_ids"].to(device)
    source_mask = src["attention_mask"].to(device).bool()

    PLACEHOLDER = "___"
    parts = partial.split(PLACEHOLDER)

    known_ids = []
    infill_positions = []
    known_ids.append(tokenizer.cls_token_id)
    infill_positions.append(False)

    for i, part in enumerate(parts):
        if part.strip():
            toks = tokenizer.encode(part.strip(), add_special_tokens=False)
            known_ids.extend(toks)
            infill_positions.extend([False] * len(toks))

        if i < len(parts) - 1:
            span_len = 5
            known_ids.extend([config.mask_token_id] * span_len)
            infill_positions.extend([True] * span_len)

    known_ids.append(tokenizer.sep_token_id)
    infill_positions.append(False)

    seq_len = config.max_seq_len
    pad_len = seq_len - len(known_ids)
    known_ids.extend([tokenizer.pad_token_id] * pad_len)
    infill_positions.extend([False] * pad_len)

    known_ids = known_ids[:seq_len]
    infill_positions = infill_positions[:seq_len]

    known_ids_t = torch.tensor([known_ids], device=device)
    infill_mask = torch.tensor([infill_positions], device=device)
    # target_mask: real tokens (not padding)
    target_mask = torch.tensor([[i < (seq_len - pad_len) for i in range(seq_len)]],
                               device=device, dtype=torch.bool)

    print(f"Template ({infill_mask.sum().item()} positions to fill):")
    preview = tokenizer.decode(known_ids_t[0], skip_special_tokens=True)
    print(f"  {preview}")

    model.eval()
    output_ids = diffusion.p_sample_loop_infill(
        model, source_ids, source_mask,
        known_ids_t, infill_mask, target_mask,
    )

    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True, help="Source text (English)")
    parser.add_argument("--partial", type=str, default=None,
                        help="Partial target with ___ for blanks")
    args = parser.parse_args()

    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    model = DiffusionTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=0.0,
        max_seq_len=config.max_seq_len * 2,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    step = checkpoint["step"]
    del checkpoint
    print(f"Loaded checkpoint from step {step}")

    diffusion = MaskDiffusion(
        timesteps=config.timesteps,
        mask_token_id=config.mask_token_id,
        schedule=config.schedule,
    ).to(device)

    if args.partial:
        output = infill(args.text, args.partial, model, diffusion, tokenizer, config, device)
    else:
        output = translate(args.text, model, diffusion, tokenizer, config, device)

    print(f"\nSource: {args.text}")
    if args.partial:
        print(f"Partial: {args.partial}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
