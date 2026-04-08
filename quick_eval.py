"""Quick eval: translate a few sentences on CPU with minimal steps."""
import torch
from config import Config
from diffusion import MaskDiffusion
from dataset import TranslationDataset
from translate import build_model
from transformers import AutoTokenizer

config = Config()
checkpoint = torch.load("checkpoints_v31/model_step_50000.pt", map_location="cpu", weights_only=False)
if "config" in checkpoint:
    ckpt_config = checkpoint["config"]
    for attr in ["model_dim", "embed_dim", "num_heads", "num_layers", "ff_dim",
                  "max_seq_len", "timesteps", "schedule", "mask_token_id", "diffusion_type", "self_cond"]:
        if hasattr(ckpt_config, attr):
            setattr(config, attr, getattr(ckpt_config, attr))

device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
model = build_model(config, device)
model.load_state_dict(checkpoint["model"], strict=False)
model.eval()
step = checkpoint["step"]
del checkpoint
print(f"Loaded step {step}, using {config.num_layers} layers, {config.model_dim}d")

diffusion = MaskDiffusion(timesteps=config.timesteps, mask_token_id=config.mask_token_id, schedule=config.schedule)

ds = TranslationDataset("data/wmt14_en_de_tokenized_test")
# Just do 5 sentences, 1 at a time, 25 steps
N = 5
for i in range(N):
    item = ds[i]
    src_ids = item["source_ids"].unsqueeze(0)
    src_mask = item["source_mask"].unsqueeze(0)
    tgt_ids = item["target_ids"].unsqueeze(0)
    tgt_mask = item["target_mask"].unsqueeze(0)
    
    src_len = src_mask.sum().item()
    est_len = min(int(src_len * 1.1), config.max_seq_len)
    tgt_m = torch.zeros(1, config.max_seq_len, dtype=torch.bool)
    tgt_m[0, :est_len] = True
    
    with torch.no_grad():
        out = diffusion.p_sample_loop(model, src_ids, src_mask, tgt_m, num_steps=25, temperature=2.0)
    
    hyp = tokenizer.decode(out[0], skip_special_tokens=True)
    ref = tokenizer.decode(tgt_ids[0][tgt_mask[0].bool()], skip_special_tokens=True)
    src = tokenizer.decode(src_ids[0][src_mask[0].bool()], skip_special_tokens=True)
    print(f"\n[{i+1}] SRC: {src}")
    print(f"    REF: {ref}")
    print(f"    HYP: {hyp}")
