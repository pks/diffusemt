"""Initialize a deeper model from a v25 checkpoint by layer stacking.

Strategy: for a 2× deeper model (e.g. 12→24 layers), repeat the layer stack.
Layer i of the new model gets the weights of layer (i % 12) of the source model.
This is equivalent to running the same 12-layer transformer twice — a valid
initialization that preserves the learned representations.
"""
import argparse
import torch
import copy
from config import Config
from model import SourceCorruptionEncoderOnly


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--num-layers", type=int, default=24)
    args = parser.parse_args()

    config = Config()

    # Load source checkpoint
    print(f"Loading source checkpoint: {args.source_checkpoint}")
    ckpt = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    src_state = ckpt["model"]
    src_step = ckpt["step"]

    # Infer source layer count from checkpoint keys (don't rely on config)
    src_layers = max(int(k.split(".")[1]) for k in src_state if k.startswith("layers.")) + 1
    print(f"Source step: {src_step}, source layers: {src_layers} → target layers: {args.num_layers}")

    # Build new model
    new_config = copy.copy(config)
    new_config.num_layers = args.num_layers
    model = SourceCorruptionEncoderOnly(
        pretrained_name=config.pretrained_name,
        model_dim=config.model_dim,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=args.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        freeze_embeddings=config.freeze_embeddings,
    )

    new_state = model.state_dict()

    # Copy all non-layer params directly
    copied = 0
    for k in new_state:
        if not k.startswith("layers."):
            if k in src_state:
                new_state[k] = src_state[k].clone()
                copied += 1

    # Map layers: new layer i ← source layer (i % src_layers)
    for new_i in range(args.num_layers):
        src_i = new_i % src_layers
        src_prefix = f"layers.{src_i}."
        new_prefix = f"layers.{new_i}."
        layer_keys = [k for k in src_state if k.startswith(src_prefix)]
        for k in layer_keys:
            new_k = new_prefix + k[len(src_prefix):]
            new_state[new_k] = src_state[k].clone()
            copied += 1

    model.load_state_dict(new_state)
    print(f"Copied {copied} tensors")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    print(f"New model: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable, {frozen/1e6:.1f}M frozen")

    # Save new checkpoint (no optimizer state — fresh training start)
    out = {
        "model": model.state_dict(),
        "step": 0,
        "config": vars(new_config),
        "source_checkpoint": args.source_checkpoint,
        "source_step": src_step,
        "init_strategy": f"layer_stack_{src_layers}to{args.num_layers}",
    }
    torch.save(out, args.output_checkpoint)
    print(f"Saved: {args.output_checkpoint}")


if __name__ == "__main__":
    main()
