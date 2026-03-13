"""Absorbing-state discrete diffusion for text.

Forward process: gradually replace target tokens with [MASK].
Source tokens are never corrupted — they provide context.
Reverse process: iteratively predict and unmask target tokens.
"""
import math
import torch
import torch.nn as nn


class MaskDiffusion(nn.Module):
    """Absorbing-state (mask) diffusion over target tokens only.

    Forward: each target token is independently replaced with [MASK] with
    probability gamma(t), where gamma(0) ≈ 0 and gamma(T) = 1.

    Reverse: model predicts original tokens from [source | masked_target],
    iteratively unmasking by confidence.
    """

    def __init__(self, timesteps=200, mask_token_id=103, schedule="cosine"):
        super().__init__()
        self.timesteps = timesteps
        self.mask_token_id = mask_token_id

        # gamma(t) = probability that each target token is masked at timestep t
        if schedule == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            gamma = 1.0 - torch.cos(math.pi / 2 * steps / timesteps) ** 2
        elif schedule == "linear":
            gamma = torch.linspace(0.0, 1.0, timesteps + 1, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        gamma = gamma.clamp(min=0.0, max=1.0).float()
        self.register_buffer("gamma", gamma)

    def q_sample(self, target_ids, t):
        """Forward process: mask target tokens independently.

        Args:
            target_ids: (B, T) clean target token IDs
            t: (B,) timestep indices (1..T)

        Returns:
            corrupted_ids: (B, T) target with some tokens replaced by [MASK]
            is_masked: (B, T) bool, True = this position was masked
        """
        B, T = target_ids.shape
        gamma_t = self.gamma[t]  # (B,)

        rand = torch.rand(B, T, device=target_ids.device)
        is_masked = rand < gamma_t[:, None]

        corrupted = target_ids.clone()
        corrupted[is_masked] = self.mask_token_id

        return corrupted, is_masked

    def _build_input(self, source_ids, source_mask, target_ids, target_mask):
        """Concatenate [source | target] with segment IDs and padding mask.

        Returns:
            input_ids: (B, S+T)
            padding_mask: (B, S+T)
            segment_ids: (B, S+T) — 0 for source, 1 for target
            src_len: int (S)
        """
        B = source_ids.shape[0]
        S = source_ids.shape[1]
        T = target_ids.shape[1]

        input_ids = torch.cat([source_ids, target_ids], dim=1)
        padding_mask = torch.cat([source_mask, target_mask], dim=1)

        segment_ids = torch.zeros(B, S + T, device=source_ids.device, dtype=torch.long)
        segment_ids[:, S:] = 1

        return input_ids, padding_mask, segment_ids, S

    @torch.no_grad()
    def p_sample_loop(self, model, source_ids, source_mask, target_len,
                      pad_token_id=0, cls_token_id=101, sep_token_id=102):
        """Full reverse process: start with all-[MASK] target, iteratively unmask.

        Args:
            model: DiffusionTransformer
            source_ids: (B, S) source token IDs
            source_mask: (B, S) bool mask
            target_len: int, length of target sequence (typically max_seq_len)
            pad_token_id: token ID for padding
            cls_token_id: [CLS] token ID
            sep_token_id: [SEP] token ID
        """
        B = source_ids.shape[0]
        S = source_ids.shape[1]
        device = source_ids.device

        # Estimate target length from source (target ≈ source * 1.5, capped)
        src_lens = source_mask.sum(dim=-1)  # (B,)
        est_target_lens = (src_lens.float() * 1.5).long().clamp(min=5, max=target_len)

        # Start: [CLS] ... [MASK] ... [SEP] [PAD] ... [PAD]
        target_ids = torch.full((B, target_len), pad_token_id,
                                device=device, dtype=torch.long)
        target_mask = torch.zeros(B, target_len, device=device, dtype=torch.bool)
        for b in range(B):
            tlen = est_target_lens[b].item()
            target_ids[b, 0] = cls_token_id
            target_ids[b, 1:tlen-1] = self.mask_token_id
            target_ids[b, tlen-1] = sep_token_id
            target_mask[b, :tlen] = True

        # Track which positions are generatable (not CLS, SEP, or PAD)
        gen_mask = (target_ids == self.mask_token_id)  # initially, only MASK positions

        for t in reversed(range(1, self.timesteps + 1)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            # Build concatenated input
            input_ids, padding_mask, segment_ids, _ = self._build_input(
                source_ids, source_mask, target_ids, target_mask)

            # Model predicts logits for all positions
            logits = model(input_ids, padding_mask, segment_ids, t_tensor)

            # Only care about target positions (after source)
            target_logits = logits[:, S:]  # (B, T, V)
            probs = torch.softmax(target_logits, dim=-1)
            pred_tokens = probs.argmax(dim=-1)  # (B, T)
            confidence = probs.max(dim=-1).values  # (B, T)

            # How many generatable tokens should still be masked at t-1?
            gamma_prev = self.gamma[t - 1]
            n_gen = gen_mask.sum(dim=-1).float()  # per-batch generatable count
            n_mask_prev = (gamma_prev * n_gen).long().clamp(min=0)  # (B,)

            if t > 1:
                is_masked = (target_ids == self.mask_token_id)

                # Only sort within generatable positions
                confidence_for_sort = confidence.clone()
                confidence_for_sort[~gen_mask] = float('inf')
                confidence_for_sort[~is_masked] = float('inf')

                sorted_idx = confidence_for_sort.argsort(dim=-1)

                new_target = target_ids.clone()
                # Unmask: replace MASK with predictions
                new_target[is_masked] = pred_tokens[is_masked]
                # Re-mask least confident
                for b in range(B):
                    keep_masked = sorted_idx[b, :n_mask_prev[b]]
                    new_target[b, keep_masked] = self.mask_token_id
                target_ids = new_target
            else:
                # Final step: unmask everything
                is_masked = (target_ids == self.mask_token_id)
                target_ids[is_masked] = pred_tokens[is_masked]

        return target_ids

    @torch.no_grad()
    def p_sample_loop_infill(self, model, source_ids, source_mask,
                             known_ids, infill_mask, target_mask):
        """Infilling: keep known target positions fixed, generate unknown ones.

        Args:
            known_ids: (B, T) token IDs (known positions have real tokens)
            infill_mask: (B, T) bool, True = positions to generate
            target_mask: (B, T) bool, True = real (non-padding) positions
        """
        B = source_ids.shape[0]
        S = source_ids.shape[1]
        device = source_ids.device
        T = known_ids.shape[1]

        # Start: known positions keep their tokens, unknown are [MASK]
        target_ids = known_ids.clone()
        target_ids[infill_mask] = self.mask_token_id

        for t in reversed(range(1, self.timesteps + 1)):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            input_ids, padding_mask, segment_ids, _ = self._build_input(
                source_ids, source_mask, target_ids, target_mask)

            logits = model(input_ids, padding_mask, segment_ids, t_tensor)
            target_logits = logits[:, S:]
            probs = torch.softmax(target_logits, dim=-1)
            pred_tokens = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values

            # Only operate on infill positions
            gamma_prev = self.gamma[t - 1]
            n_infill = infill_mask.sum(dim=-1).float()
            n_mask_prev = (gamma_prev * n_infill).long().clamp(min=0)

            if t > 1:
                new_target = target_ids.clone()
                new_target[infill_mask] = pred_tokens[infill_mask]

                # Re-mask least confident infill positions
                confidence_infill = confidence.clone()
                confidence_infill[~infill_mask] = float('inf')

                sorted_idx = confidence_infill.argsort(dim=-1)
                for b in range(B):
                    keep_masked = sorted_idx[b, :n_mask_prev[b]]
                    new_target[b, keep_masked] = self.mask_token_id

                # Always keep known positions fixed
                new_target[~infill_mask] = known_ids[~infill_mask]
                target_ids = new_target
            else:
                target_ids[infill_mask] = pred_tokens[infill_mask]

        return target_ids
