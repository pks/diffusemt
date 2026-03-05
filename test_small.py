"""Small-scale sanity check: overfit a tiny model on a few sentences.

Proves the diffusion training loop works end-to-end with the emb_scale fix.
Should see loss drop well below 1.0 and produce recognizable translations
for the memorized sentences.
"""
import torch
from torch.utils.data import DataLoader, Subset
from model import DiffusionTransformer
from diffusion import GaussianDiffusion
from dataset import TranslationDataset
from translate import translate
from transformers import AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tiny model config
EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 4
FF_DIM = 512
MAX_SEQ_LEN = 128
TIMESTEPS = 200
BATCH_SIZE = 16
NUM_STEPS = 2000
LOG_EVERY = 100


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Small subset: first 100 training examples
    full_ds = TranslationDataset("data/wmt14_en_de_tokenized")
    ds = Subset(full_ds, range(100))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = DiffusionTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=0.0,
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model params: {param_count / 1e6:.1f}M")

    diffusion = GaussianDiffusion(
        timesteps=TIMESTEPS, beta_start=1e-4, beta_end=0.02,
    ).to(DEVICE)

    # Fixed emb_scale: per-dimension std (should be ~1.0)
    with torch.no_grad():
        emb_scale = model.token_embedding.weight.std().item()
    print(f"emb_scale (std): {emb_scale:.4f}")

    # Also show what the OLD broken scale would be
    with torch.no_grad():
        old_scale = model.token_embedding.weight.norm(dim=-1).mean().item()
    print(f"old emb_scale (L2 norm): {old_scale:.4f}  <-- would crush signal")

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    # Training
    model.train()
    step = 0
    running_loss = 0.0

    while step < NUM_STEPS:
        for batch in loader:
            if step >= NUM_STEPS:
                break

            source_ids = batch["source_ids"].to(DEVICE)
            source_mask = batch["source_mask"].to(DEVICE)
            target_ids = batch["target_ids"].to(DEVICE)
            target_mask = batch["target_mask"].to(DEVICE)

            with torch.no_grad():
                x0 = model.token_embedding(target_ids) / emb_scale

            B = x0.shape[0]
            t = torch.randint(0, TIMESTEPS, (B,), device=DEVICE)
            xt, noise = diffusion.q_sample(x0, t)

            # Self-conditioning 50% of the time
            x0_self_cond = None
            if torch.rand(1).item() > 0.5:
                with torch.no_grad():
                    x0_self_cond = model(source_ids, source_mask, xt, t).detach()

            predicted_x0 = model(source_ids, source_mask, xt, t, x0_self_cond=x0_self_cond)

            mask = target_mask.unsqueeze(-1).float()
            loss = ((predicted_x0 - x0) ** 2 * mask).sum() / mask.sum() / EMBED_DIM
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % LOG_EVERY == 0:
                avg = running_loss / LOG_EVERY
                print(f"Step {step}/{NUM_STEPS} | Loss: {avg:.4f}")
                running_loss = 0.0

    # Test: translate the first few training sentences (should overfit)
    print("\n--- Overfit check (translating training sentences) ---")
    model.eval()

    class TinyConfig:
        max_seq_len = MAX_SEQ_LEN
        embed_dim = EMBED_DIM
        timesteps = TIMESTEPS
        beta_start = 1e-4
        beta_end = 0.02

    for i in range(5):
        item = full_ds[i]
        src_text = tokenizer.decode(item["source_ids"], skip_special_tokens=True)
        ref_text = tokenizer.decode(item["target_ids"], skip_special_tokens=True)
        out_text = translate(src_text, model, diffusion, tokenizer, TinyConfig(), DEVICE, emb_scale)
        print(f"\nSRC: {src_text[:100]}")
        print(f"REF: {ref_text[:100]}")
        print(f"OUT: {out_text[:100]}")


if __name__ == "__main__":
    main()
