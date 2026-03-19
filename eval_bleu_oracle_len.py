"""Evaluate BLEU with oracle target lengths (actual reference length).

This measures the upper bound if we had a perfect length predictor.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from config import Config
from diffusion import SourceCorruptionDiffusion, MaskDiffusion
from dataset import TranslationDataset
from translate import build_model
from transformers import AutoTokenizer
import sacrebleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--length-ratio", type=float, default=None,
                        help="If set, use src_len * ratio instead of oracle")
    args = parser.parse_args()

    config = Config()
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    # Infer actual layer count from checkpoint (handles mismatches with current config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    layer_keys = [k for k in checkpoint["model"] if k.startswith("layers.")]
    if layer_keys:
        config.num_layers = max(int(k.split(".")[1]) for k in layer_keys) + 1
    model = build_model(config, device)
    model.load_state_dict(checkpoint["model"])
    step = checkpoint["step"]
    del checkpoint
    print(f"Loaded checkpoint from step {step}")

    mode = f"oracle" if args.length_ratio is None else f"ratio={args.length_ratio}"
    print(f"Length mode: {mode}, temperature={args.temperature}, num_steps={args.num_steps or config.timesteps}")

    diffusion_cls = MaskDiffusion if getattr(config, 'diffusion_type', 'source') == 'mask' else SourceCorruptionDiffusion
    diffusion = diffusion_cls(
        timesteps=config.timesteps,
        mask_token_id=config.mask_token_id,
        schedule=config.schedule,
    ).to(device)

    model.eval()
    ds = TranslationDataset("data/wmt14_en_de_tokenized_test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    hypotheses = []
    references = []
    n_done = 0

    hyp_lens = []
    ref_lens = []

    for batch in loader:
        if n_done >= args.n:
            break

        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        B = source_ids.shape[0]

        if args.length_ratio is not None:
            # Heuristic ratio
            src_lens = source_mask.sum(dim=-1)
            est_tgt_lens = (src_lens.float() * args.length_ratio).long().clamp(min=1, max=config.max_seq_len)
        else:
            # Oracle: use actual reference length
            est_tgt_lens = target_mask.sum(dim=-1).clamp(max=config.max_seq_len)

        tgt_masks = torch.zeros(B, config.max_seq_len, device=device, dtype=torch.bool)
        for i in range(B):
            tgt_masks[i, :est_tgt_lens[i]] = True

        with torch.no_grad():
            output_ids = diffusion.p_sample_loop(
                model, source_ids, source_mask, tgt_masks,
                num_steps=args.num_steps, temperature=args.temperature)

        for i in range(B):
            if n_done >= args.n:
                break

            hyp = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            ref = tokenizer.decode(target_ids[i][target_mask[i].bool()], skip_special_tokens=True)

            hypotheses.append(hyp)
            references.append(ref)
            hyp_lens.append(len(hyp.split()))
            ref_lens.append(len(ref.split()))
            n_done += 1

        print(f"  [{n_done}/{args.n}]")

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    avg_hyp = sum(hyp_lens) / len(hyp_lens)
    avg_ref = sum(ref_lens) / len(ref_lens)
    print(f"\n{'='*60}")
    print(f"BLEU ({mode}): {bleu.score:.2f}")
    print(f"Details: {bleu}")
    print(f"Avg hyp len: {avg_hyp:.1f} words | Avg ref len: {avg_ref:.1f} words | ratio: {avg_hyp/avg_ref:.3f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
