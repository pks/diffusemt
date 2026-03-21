"""Evaluate BLEU score on WMT14 EN→DE test set."""
import argparse
import torch
from torch.utils.data import DataLoader
from config import Config
from diffusion import SourceCorruptionDiffusion, MaskDiffusion
from dataset import TranslationDataset
from model import LengthPredictor
from translate import build_model
from transformers import AutoTokenizer
import sacrebleu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n", type=int, default=200, help="Number of sentences to evaluate")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=2.0, help="Sampling temperature")
    parser.add_argument("--num-steps", type=int, default=None, help="Number of sampling steps (default: T)")
    parser.add_argument("--use-length-predictor", action="store_true", help="Use trained length predictor instead of 1.1x heuristic")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic sampling for non-final steps")
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
    model.load_state_dict(checkpoint["model"], strict=False)
    has_length_predictor = "length_predictor" in checkpoint
    length_predictor = None
    if has_length_predictor and args.use_length_predictor:
        length_predictor = LengthPredictor(
            word_embeddings=model.word_embeddings,
            embed_dim=config.embed_dim,
            hidden_dim=config.model_dim,
        ).to(device)
        length_predictor.load_state_dict(checkpoint["length_predictor"], strict=False)
        length_predictor.eval()
    step = checkpoint["step"]
    del checkpoint
    print(f"Loaded checkpoint from step {step}")
    print(f"Sampling: temperature={args.temperature}, num_steps={args.num_steps or config.timesteps}, "
          f"stochastic={args.stochastic}, length_predictor={'trained' if has_length_predictor else 'none'}")

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

    for batch in loader:
        if n_done >= args.n:
            break

        source_ids = batch["source_ids"].to(device)
        source_mask = batch["source_mask"].to(device)
        target_ids = batch["target_ids"]
        target_mask = batch["target_mask"]

        B = source_ids.shape[0]

        # Estimate target length per sample
        if length_predictor is not None:
            with torch.no_grad():
                est_tgt_lens = length_predictor(source_ids, source_mask).round().long().clamp(min=1, max=config.max_seq_len)
        else:
            src_lens = source_mask.sum(dim=-1)
            est_tgt_lens = (src_lens.float() * 1.1).long().clamp(max=config.max_seq_len)

        # Build per-sample target masks
        tgt_masks = torch.zeros(B, config.max_seq_len, device=device, dtype=torch.bool)
        for i in range(B):
            tgt_masks[i, :est_tgt_lens[i]] = True

        # Generate
        with torch.no_grad():
            output_ids = diffusion.p_sample_loop(
                model, source_ids, source_mask, tgt_masks,
                num_steps=args.num_steps, temperature=args.temperature,
                stochastic=args.stochastic)

        # Decode
        for i in range(B):
            if n_done >= args.n:
                break

            hyp = tokenizer.decode(output_ids[i], skip_special_tokens=True)
            ref = tokenizer.decode(target_ids[i][target_mask[i].bool()], skip_special_tokens=True)
            src = tokenizer.decode(source_ids[i][source_mask[i].bool()], skip_special_tokens=True)

            hypotheses.append(hyp)
            references.append(ref)
            n_done += 1

            if n_done <= 5:
                print(f"\n[{n_done}] SRC: {src}")
                print(f"    REF: {ref}")
                print(f"    HYP: {hyp}")

        print(f"  [{n_done}/{args.n} sentences translated]")

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    print(f"\n{'='*60}")
    print(f"BLEU on {len(hypotheses)} test sentences: {bleu.score:.2f}")
    print(f"Details: {bleu}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
