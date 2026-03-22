"""Knowledge distillation: translate EN training data with MarianMT teacher.

Replaces reference German targets with AR model outputs for cleaner training signal.
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer
from dataset import TranslationDataset
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1_000_000,
                        help="Number of sentences to distill (default: 1M)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="data/wmt14_en_de_distilled")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam size for Marian")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Teacher model (MarianMT EN→DE)
    marian_name = "Helsinki-NLP/opus-mt-en-de"
    marian_tok = MarianTokenizer.from_pretrained(marian_name)
    marian_model = MarianMTModel.from_pretrained(marian_name).to(device).half().eval()

    # Student tokenizer (mBERT)
    mbert_tok = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    max_seq_len = 128

    # Load source data
    ds = TranslationDataset("data/wmt14_en_de_tokenized")
    n = min(args.n, len(ds))
    subset = Subset(ds, range(n))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    os.makedirs(args.output_dir, exist_ok=True)

    # Process in chunks and save incrementally
    all_source_ids = []
    all_source_mask = []
    all_target_ids = []
    all_target_mask = []
    done = 0
    t0 = time.time()

    for batch in loader:
        src_ids = batch["source_ids"]
        src_mask = batch["source_mask"]
        B = src_ids.shape[0]

        # Decode source to text
        texts = []
        for i in range(B):
            text = mbert_tok.decode(src_ids[i][src_mask[i].bool()], skip_special_tokens=True)
            texts.append(text)

        # Translate with Marian
        inputs = marian_tok(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_seq_len).to(device)
        with torch.no_grad():
            outputs = marian_model.generate(**inputs, max_length=max_seq_len,
                                            num_beams=args.num_beams)

        # Decode Marian output, re-tokenize with mBERT
        for i in range(B):
            german_text = marian_tok.decode(outputs[i], skip_special_tokens=True)
            encoded = mbert_tok(german_text, max_length=max_seq_len,
                                truncation=True, padding="max_length",
                                return_tensors="pt")
            tgt_ids = encoded["input_ids"].squeeze(0)
            tgt_mask = encoded["attention_mask"].squeeze(0)

            # Strip [CLS] and [SEP] from mBERT tokenization
            # mBERT adds [CLS] at start and [SEP] at end
            # For our dataset format, we want raw tokens without special tokens
            # Find actual tokens (between [CLS] and [SEP])
            real_len = tgt_mask.sum().item()
            if real_len >= 2:
                # Remove [CLS] (101) at pos 0 and [SEP] (102) at end
                raw_ids = tgt_ids[1:real_len - 1]
                raw_len = len(raw_ids)
            else:
                raw_ids = torch.tensor([], dtype=torch.long)
                raw_len = 0

            # Pad to max_seq_len
            padded_ids = torch.zeros(max_seq_len, dtype=torch.long)
            padded_mask = torch.zeros(max_seq_len, dtype=torch.long)
            if raw_len > 0:
                clipped_len = min(raw_len, max_seq_len)
                padded_ids[:clipped_len] = raw_ids[:clipped_len]
                padded_mask[:clipped_len] = 1

            all_target_ids.append(padded_ids)
            all_target_mask.append(padded_mask)

        # Source stays the same
        all_source_ids.append(src_ids)
        all_source_mask.append(src_mask)

        done += B
        elapsed = time.time() - t0
        rate = done / elapsed
        eta = (n - done) / rate if rate > 0 else 0
        if done % (args.batch_size * 10) == 0 or done >= n:
            print(f"  [{done}/{n}] {rate:.0f} sent/s, ETA: {eta/60:.0f}m")

    # Concatenate and save as HuggingFace dataset (compatible with load_from_disk)
    from datasets import Dataset as HFDataset
    source_ids = torch.cat(all_source_ids, dim=0)[:n]
    source_mask = torch.cat(all_source_mask, dim=0)[:n]
    target_ids = torch.stack(all_target_ids)[:n]
    target_mask = torch.stack(all_target_mask)[:n]

    hf_ds = HFDataset.from_dict({
        "source_ids": source_ids.tolist(),
        "source_mask": source_mask.tolist(),
        "target_ids": target_ids.tolist(),
        "target_mask": target_mask.tolist(),
    })
    hf_ds.save_to_disk(args.output_dir)

    elapsed = time.time() - t0
    print(f"\nDone! {n} sentences distilled in {elapsed/60:.1f}m")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
