import argparse
import torch
from config import Config
from model import DiffusionTransformer
from diffusion import GaussianDiffusion
from transformers import AutoTokenizer


def embeddings_to_tokens(embeddings, embedding_weight):
    """Map continuous embeddings to nearest tokens via L2 distance."""
    dist = torch.cdist(embeddings, embedding_weight.unsqueeze(0), p=2)  # (B, T, V)
    return dist.argmin(dim=-1)


def translate(text, model, diffusion, tokenizer, config, device, emb_scale):
    """Translate a single source sentence."""
    src = tokenizer(
        text, max_length=config.max_seq_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    source_ids = src["input_ids"].to(device)
    source_mask = src["attention_mask"].to(device).bool()

    model.eval()
    # Normalized embedding weight for clamping during reverse diffusion
    norm_emb_weight = model.token_embedding.weight.data / emb_scale

    denoised = diffusion.p_sample_loop(
        model, source_ids, source_mask,
        seq_len=config.max_seq_len, embed_dim=config.embed_dim,
        embedding_weight=norm_emb_weight,
    )

    denoised = denoised * emb_scale
    embedding_weight = model.token_embedding.weight.data
    token_ids = embeddings_to_tokens(denoised, embedding_weight)
    return tokenizer.decode(token_ids[0], skip_special_tokens=True)


def infill(text, partial, model, diffusion, tokenizer, config, device, emb_scale):
    """Fill in masked spans of a partial translation.

    partial: target text with ___ for spans to generate.
             e.g. "Das ___ ist heute ___."
    """
    src = tokenizer(
        text, max_length=config.max_seq_len, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    source_ids = src["input_ids"].to(device)
    source_mask = src["attention_mask"].to(device).bool()

    # Parse partial: split on ___ placeholder, tokenize known pieces,
    # and build the known_ids + infill_mask
    PLACEHOLDER = "___"
    parts = partial.split(PLACEHOLDER)

    # Build token sequence and mask
    known_ids = []
    infill_positions = []
    # Add [CLS]
    known_ids.append(tokenizer.cls_token_id)
    infill_positions.append(False)

    for i, part in enumerate(parts):
        # Tokenize this known fragment (without special tokens)
        if part.strip():
            toks = tokenizer.encode(part.strip(), add_special_tokens=False)
            known_ids.extend(toks)
            infill_positions.extend([False] * len(toks))

        # If not the last part, insert placeholder span to infill
        if i < len(parts) - 1:
            # Use a fixed span length for unknowns (5 tokens)
            span_len = 5
            known_ids.extend([tokenizer.mask_token_id] * span_len)
            infill_positions.extend([True] * span_len)

    # Add [SEP]
    known_ids.append(tokenizer.sep_token_id)
    infill_positions.append(False)

    # Pad to max_seq_len
    seq_len = config.max_seq_len
    pad_len = seq_len - len(known_ids)
    known_ids.extend([tokenizer.pad_token_id] * pad_len)
    infill_positions.extend([False] * pad_len)

    # Truncate if needed
    known_ids = known_ids[:seq_len]
    infill_positions = infill_positions[:seq_len]

    known_ids_t = torch.tensor([known_ids], device=device)
    infill_mask = torch.tensor([infill_positions], device=device)  # True = generate

    print(f"Template ({infill_mask.sum().item()} positions to fill):")
    preview = tokenizer.decode(known_ids_t[0], skip_special_tokens=True)
    print(f"  {preview}")

    # Get known embeddings (normalized)
    model.eval()
    norm_emb_weight = model.token_embedding.weight.data / emb_scale
    with torch.no_grad():
        known_x0 = model.token_embedding(known_ids_t) / emb_scale

    # Run infilling
    denoised = diffusion.p_sample_loop_infill(
        model, source_ids, source_mask,
        known_x0, infill_mask,
        seq_len=seq_len, embed_dim=config.embed_dim,
        embedding_weight=norm_emb_weight,
    )

    denoised = denoised * emb_scale
    embedding_weight = model.token_embedding.weight.data
    token_ids = embeddings_to_tokens(denoised, embedding_weight)
    return tokenizer.decode(token_ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--text", type=str, required=True, help="Source text (English)")
    parser.add_argument("--partial", type=str, default=None,
                        help="Partial target with ___ for blanks, e.g. 'Das ___ ist heute ___.'")
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
        max_seq_len=config.max_seq_len,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    emb_scale = checkpoint.get("emb_scale", 1.0)
    step = checkpoint["step"]
    del checkpoint  # free memory
    print(f"Loaded checkpoint from step {step}, emb_scale={emb_scale:.2f}")

    diffusion = GaussianDiffusion(
        timesteps=config.timesteps,
        beta_start=config.beta_start,
        beta_end=config.beta_end,
    ).to(device)

    if args.partial:
        output = infill(args.text, args.partial, model, diffusion, tokenizer, config, device, emb_scale)
    else:
        output = translate(args.text, model, diffusion, tokenizer, config, device, emb_scale)

    print(f"\nSource: {args.text}")
    if args.partial:
        print(f"Partial: {args.partial}")
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
