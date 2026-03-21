"""Discrete diffusion for translation.

Two strategies:
1. Source-as-corruption: replace German tokens with English source tokens as noise.
2. Mask diffusion: replace German tokens with [MASK], source provided as context.

Reverse process: iteratively predict German tokens, unmasking by confidence.
"""
import math
import torch
import torch.nn as nn


class SourceCorruptionDiffusion(nn.Module):
    """Source-as-corruption diffusion.

    Forward: each target token is independently replaced with the corresponding
    source token (or [MASK] if no source token at that position) with
    probability gamma(t).

    Reverse: model predicts clean German tokens from mixed EN/DE input,
    iteratively replacing by confidence.
    """

    def __init__(self, timesteps=200, mask_token_id=103, schedule="cosine"):
        super().__init__()
        self.timesteps = timesteps
        self.mask_token_id = mask_token_id

        if schedule == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            gamma = 1.0 - torch.cos(math.pi / 2 * steps / timesteps) ** 2
        elif schedule == "linear":
            gamma = torch.linspace(0.0, 1.0, timesteps + 1, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        gamma = gamma.clamp(min=0.0, max=1.0).float()
        self.register_buffer("gamma", gamma)

    def q_sample(self, source_ids, source_mask, target_ids, target_mask, t):
        """Forward process: replace target tokens with source tokens as noise.

        Args:
            source_ids: (B, L) English source token IDs (padded to L)
            source_mask: (B, L) bool, True = real source token
            target_ids: (B, L) German target token IDs (padded to L)
            target_mask: (B, L) bool, True = real target token
            t: (B,) timestep indices (1..T)

        Returns:
            corrupted: (B, L) mixed EN/DE tokens
            is_corrupted: (B, L) bool, True = position was corrupted
        """
        B, L = target_ids.shape
        gamma_t = self.gamma[t]  # (B,)

        # Decide which positions to corrupt
        rand = torch.rand(B, L, device=target_ids.device)
        should_corrupt = rand < gamma_t[:, None]

        # Only corrupt real target positions
        should_corrupt = should_corrupt & target_mask

        corrupted = target_ids.clone()

        # Where source is available, replace with source token
        has_source = should_corrupt & source_mask
        corrupted[has_source] = source_ids[has_source]

        # Where source is NOT available (target longer than source), use [MASK]
        no_source = should_corrupt & ~source_mask
        corrupted[no_source] = self.mask_token_id

        return corrupted, should_corrupt

    def _make_fully_corrupted(self, source_ids, source_mask, target_mask):
        """Create the fully-corrupted state (t=T): source tokens + [MASK] fallback.

        Returns:
            corrupted: (B, L) — source tokens where available, [MASK] where
                       target extends beyond source, [PAD] elsewhere
            gen_mask: (B, L) bool — positions that need to be generated
        """
        B, L = source_ids.shape
        pad_token_id = 0
        corrupted = torch.full((B, L), pad_token_id,
                               device=source_ids.device, dtype=torch.long)

        # Place source tokens where source is real
        corrupted[source_mask] = source_ids[source_mask]

        # [MASK] where target is real but source is not (target longer than source)
        mask_positions = target_mask & ~source_mask
        corrupted[mask_positions] = self.mask_token_id

        # Generatable = all real target positions
        gen_mask = target_mask.clone()

        return corrupted, gen_mask

    @torch.no_grad()
    def p_sample_loop(self, model, source_ids, source_mask, target_mask,
                      num_steps=None, temperature=1.0):
        """Full reverse process: start from English source, denoise to German.

        Args:
            model: SourceCorruptionTransformer
            source_ids: (B, L) English source token IDs
            source_mask: (B, L) bool
            target_mask: (B, L) bool — which positions are real target
        """
        B = source_ids.shape[0]
        device = source_ids.device

        # Start from fully corrupted state
        current_ids, gen_mask = self._make_fully_corrupted(
            source_ids, source_mask, target_mask)

        # Pre-compute encoder output if model supports it (encoder-decoder)
        enc_out = None
        if hasattr(model, 'encode_source'):
            enc_out = model.encode_source(source_ids, source_mask)

        # Build timestep schedule (optionally skip steps for faster sampling)
        if num_steps is not None and num_steps < self.timesteps:
            # Evenly spaced timesteps from T down to 1
            step_indices = torch.linspace(self.timesteps, 1, num_steps).long().tolist()
            # Ensure we end at t=1
            if step_indices[-1] != 1:
                step_indices[-1] = 1
        else:
            step_indices = list(range(self.timesteps, 0, -1))

        for i, t in enumerate(step_indices):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            logits = model(current_ids, target_mask, t_tensor,
                           source_ids=source_ids, source_mask=source_mask,
                           encoder_output=enc_out)

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
            pred_tokens = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values

            # How many positions should still be corrupted at next timestep?
            if i + 1 < len(step_indices):
                next_t = step_indices[i + 1]
                gamma_next = self.gamma[next_t]
            else:
                gamma_next = 0.0

            n_gen = gen_mask.sum(dim=-1).float()
            n_corrupt_next = (gamma_next * n_gen).long().clamp(min=0)

            if i + 1 < len(step_indices):
                # All gen positions get predicted German tokens
                new_ids = current_ids.clone()
                new_ids[gen_mask] = pred_tokens[gen_mask]

                # Re-corrupt least confident positions with source tokens
                confidence_for_sort = confidence.clone()
                confidence_for_sort[~gen_mask] = float('inf')

                sorted_idx = confidence_for_sort.argsort(dim=-1)
                for b in range(B):
                    re_corrupt = sorted_idx[b, :n_corrupt_next[b]]
                    # Put back source token or [MASK]
                    for idx in re_corrupt:
                        if source_mask[b, idx]:
                            new_ids[b, idx] = source_ids[b, idx]
                        else:
                            new_ids[b, idx] = self.mask_token_id

                current_ids = new_ids
            else:
                # Final step: predict everything
                current_ids[gen_mask] = pred_tokens[gen_mask]

        return current_ids

    @torch.no_grad()
    def p_sample_loop_infill(self, model, source_ids, source_mask,
                             known_ids, infill_mask, target_mask,
                             num_steps=None, temperature=1.0):
        """Infilling: keep known German positions fixed, denoise the rest.

        Args:
            known_ids: (B, L) token IDs (known positions have real German tokens)
            infill_mask: (B, L) bool, True = positions to generate
            target_mask: (B, L) bool, True = real (non-padding) positions
        """
        B = source_ids.shape[0]
        device = source_ids.device

        # Start: known positions keep German tokens, infill positions get source/[MASK]
        current_ids = known_ids.clone()
        for b in range(B):
            for i in range(current_ids.shape[1]):
                if infill_mask[b, i]:
                    if source_mask[b, i]:
                        current_ids[b, i] = source_ids[b, i]
                    else:
                        current_ids[b, i] = self.mask_token_id

        # Pre-compute encoder output if model supports it
        enc_out = None
        if hasattr(model, 'encode_source'):
            enc_out = model.encode_source(source_ids, source_mask)

        # Build timestep schedule
        if num_steps is not None and num_steps < self.timesteps:
            step_indices = torch.linspace(self.timesteps, 1, num_steps).long().tolist()
            if step_indices[-1] != 1:
                step_indices[-1] = 1
        else:
            step_indices = list(range(self.timesteps, 0, -1))

        for i, t in enumerate(step_indices):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            logits = model(current_ids, target_mask, t_tensor,
                           source_ids=source_ids, source_mask=source_mask,
                           encoder_output=enc_out)

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
            pred_tokens = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values

            if i + 1 < len(step_indices):
                next_t = step_indices[i + 1]
                gamma_next = self.gamma[next_t]
            else:
                gamma_next = 0.0

            n_infill = infill_mask.sum(dim=-1).float()
            n_corrupt_next = (gamma_next * n_infill).long().clamp(min=0)

            if i + 1 < len(step_indices):
                new_ids = current_ids.clone()
                new_ids[infill_mask] = pred_tokens[infill_mask]

                # Re-corrupt least confident infill positions
                confidence_infill = confidence.clone()
                confidence_infill[~infill_mask] = float('inf')
                sorted_idx = confidence_infill.argsort(dim=-1)

                for b in range(B):
                    re_corrupt = sorted_idx[b, :n_corrupt_next[b]]
                    for idx in re_corrupt:
                        if source_mask[b, idx]:
                            new_ids[b, idx] = source_ids[b, idx]
                        else:
                            new_ids[b, idx] = self.mask_token_id

                # Keep known positions fixed
                new_ids[~infill_mask] = known_ids[~infill_mask]
                current_ids = new_ids
            else:
                current_ids[infill_mask] = pred_tokens[infill_mask]

        return current_ids


class MaskDiffusion(nn.Module):
    """Mask-based discrete diffusion for translation.

    Forward: each target token is independently replaced with [MASK] with
    probability gamma(t). Source tokens are provided as context, not noise.

    Reverse: model predicts clean German tokens from masked input,
    iteratively unmasking by confidence.
    """

    def __init__(self, timesteps=200, mask_token_id=103, schedule="cosine"):
        super().__init__()
        self.timesteps = timesteps
        self.mask_token_id = mask_token_id

        if schedule == "cosine":
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            gamma = 1.0 - torch.cos(math.pi / 2 * steps / timesteps) ** 2
        elif schedule == "linear":
            gamma = torch.linspace(0.0, 1.0, timesteps + 1, dtype=torch.float64)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

        gamma = gamma.clamp(min=0.0, max=1.0).float()
        self.register_buffer("gamma", gamma)

    def q_sample(self, source_ids, source_mask, target_ids, target_mask, t):
        """Forward process: replace target tokens with [MASK].

        Args:
            source_ids: (B, L) English source token IDs (unused for corruption, kept for API compat)
            source_mask: (B, L) bool (unused for corruption)
            target_ids: (B, L) German target token IDs
            target_mask: (B, L) bool, True = real target token
            t: (B,) timestep indices (1..T)

        Returns:
            corrupted: (B, L) tokens with some positions replaced by [MASK]
            is_corrupted: (B, L) bool, True = position was masked
        """
        B, L = target_ids.shape
        gamma_t = self.gamma[t]  # (B,)

        rand = torch.rand(B, L, device=target_ids.device)
        should_mask = rand < gamma_t[:, None]
        should_mask = should_mask & target_mask

        corrupted = target_ids.clone()
        corrupted[should_mask] = self.mask_token_id

        return corrupted, should_mask

    @torch.no_grad()
    def p_sample_loop(self, model, source_ids, source_mask, target_mask,
                      num_steps=None, temperature=1.0, stochastic=False):
        """Full reverse process: start from all [MASK], denoise to German.

        Args:
            stochastic: if True, sample from distribution on non-final steps
                        (adds diversity; least-confident positions get re-masked)
        """
        B = source_ids.shape[0]
        L = target_mask.shape[1]
        device = source_ids.device

        # Start from fully masked state
        current_ids = torch.full((B, L), self.mask_token_id,
                                 device=device, dtype=torch.long)
        current_ids[~target_mask] = 0
        gen_mask = target_mask.clone()

        enc_out = None
        if hasattr(model, 'encode_source'):
            enc_out = model.encode_source(source_ids, source_mask)

        if num_steps is not None and num_steps < self.timesteps:
            step_indices = torch.linspace(self.timesteps, 1, num_steps).long().tolist()
            if step_indices[-1] != 1:
                step_indices[-1] = 1
        else:
            step_indices = list(range(self.timesteps, 0, -1))

        for i, t in enumerate(step_indices):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            logits = model(current_ids, target_mask, t_tensor,
                           source_ids=source_ids, source_mask=source_mask,
                           encoder_output=enc_out)

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)

            is_final = (i + 1 >= len(step_indices))
            if stochastic and not is_final:
                # Sample from distribution (non-final steps)
                flat_probs = probs.view(-1, probs.shape[-1])
                sampled = torch.multinomial(flat_probs, 1).squeeze(-1)
                pred_tokens = sampled.view(B, L)
                confidence = flat_probs.gather(1, sampled.unsqueeze(1)).squeeze(1).view(B, L)
            else:
                # Argmax (final step or deterministic mode)
                pred_tokens = probs.argmax(dim=-1)
                confidence = probs.max(dim=-1).values

            if not is_final:
                next_t = step_indices[i + 1]
                gamma_next = self.gamma[next_t]

                n_gen = gen_mask.sum(dim=-1).float()
                n_mask_next = (gamma_next * n_gen).long().clamp(min=0)

                new_ids = current_ids.clone()
                new_ids[gen_mask] = pred_tokens[gen_mask]

                # Re-mask least confident positions
                confidence_for_sort = confidence.clone()
                confidence_for_sort[~gen_mask] = float('inf')
                sorted_idx = confidence_for_sort.argsort(dim=-1)
                for b in range(B):
                    re_mask = sorted_idx[b, :n_mask_next[b]]
                    new_ids[b, re_mask] = self.mask_token_id

                current_ids = new_ids
            else:
                current_ids[gen_mask] = pred_tokens[gen_mask]

        return current_ids

    @torch.no_grad()
    def p_sample_loop_infill(self, model, source_ids, source_mask,
                             known_ids, infill_mask, target_mask,
                             num_steps=None, temperature=1.0):
        """Infilling: keep known positions fixed, denoise masked positions."""
        B = source_ids.shape[0]
        device = source_ids.device

        current_ids = known_ids.clone()
        current_ids[infill_mask] = self.mask_token_id

        enc_out = None
        if hasattr(model, 'encode_source'):
            enc_out = model.encode_source(source_ids, source_mask)

        if num_steps is not None and num_steps < self.timesteps:
            step_indices = torch.linspace(self.timesteps, 1, num_steps).long().tolist()
            if step_indices[-1] != 1:
                step_indices[-1] = 1
        else:
            step_indices = list(range(self.timesteps, 0, -1))

        for i, t in enumerate(step_indices):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

            logits = model(current_ids, target_mask, t_tensor,
                           source_ids=source_ids, source_mask=source_mask,
                           encoder_output=enc_out)

            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=-1)
            pred_tokens = probs.argmax(dim=-1)
            confidence = probs.max(dim=-1).values

            if i + 1 < len(step_indices):
                next_t = step_indices[i + 1]
                gamma_next = self.gamma[next_t]
            else:
                gamma_next = 0.0

            n_infill = infill_mask.sum(dim=-1).float()
            n_mask_next = (gamma_next * n_infill).long().clamp(min=0)

            if i + 1 < len(step_indices):
                new_ids = current_ids.clone()
                new_ids[infill_mask] = pred_tokens[infill_mask]

                confidence_infill = confidence.clone()
                confidence_infill[~infill_mask] = float('inf')
                sorted_idx = confidence_infill.argsort(dim=-1)

                for b in range(B):
                    re_mask = sorted_idx[b, :n_mask_next[b]]
                    new_ids[b, re_mask] = self.mask_token_id

                new_ids[~infill_mask] = known_ids[~infill_mask]
                current_ids = new_ids
            else:
                current_ids[infill_mask] = pred_tokens[infill_mask]

        return current_ids
