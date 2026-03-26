"""Initialize v32 (32-layer) checkpoint from v26 (24-layer) via layer stacking.

Copies all 24 layers from v26, then duplicates layers 0-7 as layers 24-31.
This gives the new layers a warm start from trained weights.
"""
import torch
import copy

src_path = "checkpoints_v26_deeper/model_step_190000.pt"
dst_path = "checkpoints_v32_init.pt"

print(f"Loading {src_path}...")
ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
state = ckpt["model"]

# Count existing layers
layer_keys = [k for k in state if k.startswith("layers.")]
n_existing = max(int(k.split(".")[1]) for k in layer_keys) + 1
print(f"Source has {n_existing} layers")

# Stack: duplicate layers 0-7 as layers 24-31
n_new = 32
for new_idx in range(n_existing, n_new):
    src_idx = new_idx - n_existing  # 0-7
    print(f"  Copying layer {src_idx} -> layer {new_idx}")
    for key in list(state.keys()):
        if key.startswith(f"layers.{src_idx}."):
            new_key = key.replace(f"layers.{src_idx}.", f"layers.{new_idx}.")
            state[new_key] = state[key].clone()

# Verify
new_layer_keys = [k for k in state if k.startswith("layers.")]
n_final = max(int(k.split(".")[1]) for k in new_layer_keys) + 1
print(f"Output has {n_final} layers")

# Save with source step info
save_dict = {
    "step": 0,
    "source_step": ckpt.get("step", 190000),
    "model": state,
}
torch.save(save_dict, dst_path)
print(f"Saved to {dst_path}")
