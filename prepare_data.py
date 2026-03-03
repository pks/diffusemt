"""Pre-tokenize dataset and save to disk. Run once before training."""
from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config


def prepare_split(config, tokenizer, split, out_path):
    ds = load_dataset(config.dataset_name, config.dataset_config, split=split)
    print(f"Loaded {len(ds)} examples for split={split}")

    # Extract text
    ds = ds.map(
        lambda ex: {"src": ex["translation"][config.src_lang], "tgt": ex["translation"][config.tgt_lang]},
        remove_columns=["translation"],
        num_proc=8, desc=f"Extracting text ({split})",
    )

    # Tokenize
    def tokenize(batch):
        src = tokenizer(batch["src"], max_length=config.max_seq_len, padding="max_length", truncation=True)
        tgt = tokenizer(batch["tgt"], max_length=config.max_seq_len, padding="max_length", truncation=True)
        return {
            "source_ids": src["input_ids"], "source_mask": src["attention_mask"],
            "target_ids": tgt["input_ids"], "target_mask": tgt["attention_mask"],
        }

    ds = ds.map(tokenize, batched=True, batch_size=10000, remove_columns=["src", "tgt"],
                num_proc=8, desc=f"Tokenizing ({split})")

    ds.save_to_disk(out_path)
    print(f"Saved {split} to {out_path}")


def main():
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    prepare_split(config, tokenizer, "train", "data/wmt14_en_de_tokenized")
    prepare_split(config, tokenizer, "test", "data/wmt14_en_de_tokenized_test")


if __name__ == "__main__":
    main()
