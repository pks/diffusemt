import torch
from config import Config
from model import DiffusionTransformer
from diffusion import GaussianDiffusion
from transformers import AutoTokenizer

config = Config()
device = torch.device("cuda:0")
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

model = DiffusionTransformer(
    vocab_size=tokenizer.vocab_size, embed_dim=config.embed_dim,
    num_heads=config.num_heads, num_layers=config.num_layers,
    ff_dim=config.ff_dim, dropout=0.0, max_seq_len=config.max_seq_len,
).to(device)

ckpt = torch.load("checkpoints/model_step_100000.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
emb_scale = ckpt.get("emb_scale", 1.0)
del ckpt

model.eval()
emb_weight = model.token_embedding.weight.data  # (V, D)
norm_emb = emb_weight / emb_scale

# Check: what token is nearest to the zero vector?
zero_dist = torch.norm(norm_emb, dim=-1)  # distance of each token from origin
nearest_zero = zero_dist.argmin()
print(f"Token nearest to zero: '{tokenizer.decode([nearest_zero])}' (id={nearest_zero.item()}, dist={zero_dist[nearest_zero]:.4f})")

# Check: what's "gerekend"?
gerekend_ids = tokenizer.encode("gerekend", add_special_tokens=False)
print(f"'gerekend' token ids: {gerekend_ids}")
for tid in gerekend_ids:
    print(f"  id={tid} '{tokenizer.decode([tid])}' norm={norm_emb[tid].norm():.4f}")

# Check: what does the model predict from pure noise at various timesteps?
src = tokenizer("The weather is nice today.", max_length=config.max_seq_len, 
                padding="max_length", truncation=True, return_tensors="pt")
source_ids = src["input_ids"].to(device)
source_mask = src["attention_mask"].to(device).bool()

with torch.no_grad():
    xt = torch.randn(1, config.max_seq_len, config.embed_dim, device=device)
    
    for t_val in [199, 150, 100, 50, 10, 0]:
        t = torch.full((1,), t_val, device=device, dtype=torch.long)
        pred = model(source_ids, source_mask, xt, t)
        
        # Map to tokens
        pred_scaled = pred * emb_scale
        dist = torch.cdist(pred_scaled, emb_weight.unsqueeze(0), p=2)
        token_ids = dist.argmin(dim=-1)
        text = tokenizer.decode(token_ids[0][:20], skip_special_tokens=False)
        
        # Check diversity
        unique_tokens = token_ids[0][:20].unique().numel()
        pred_norm = pred.norm(dim=-1).mean().item()
        
        print(f"\nt={t_val}: pred_norm={pred_norm:.4f}, unique_tokens(first20)={unique_tokens}")
        print(f"  Decoded: {text}")

# Also check: what does the model predict when given CLEAN target embeddings (t=0)?
print("\n--- Prediction from clean embeddings (should be near-identity) ---")
tgt = tokenizer("Das Wetter ist heute schön.", max_length=config.max_seq_len,
                padding="max_length", truncation=True, return_tensors="pt")
tgt_ids = tgt["input_ids"].to(device)
with torch.no_grad():
    clean_emb = model.token_embedding(tgt_ids) / emb_scale
    t = torch.full((1,), 0, device=device, dtype=torch.long)
    pred = model(source_ids, source_mask, clean_emb, t)
    pred_scaled = pred * emb_scale
    dist = torch.cdist(pred_scaled, emb_weight.unsqueeze(0), p=2)
    token_ids = dist.argmin(dim=-1)
    ref_text = tokenizer.decode(tgt_ids[0], skip_special_tokens=True)
    pred_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    print(f"  Reference: {ref_text}")
    print(f"  Predicted: {pred_text}")
