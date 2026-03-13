import math
import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 schedule="cosine"):
        super().__init__()
        self.timesteps = timesteps

        if schedule == "cosine":
            # Cosine schedule: alpha_bar(t) = cos(pi/2 * t/T)^2
            steps = torch.arange(timesteps + 1, dtype=torch.float64)
            alpha_bar = torch.cos(math.pi / 2 * steps / timesteps) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar[0] = 1
            # Derive betas from alpha_bar, clip to prevent singularities
            betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
            betas = betas.clamp(max=0.999).float()
            alpha_bar = alpha_bar[1:].float()  # drop the t=0 entry
            alphas = (1.0 - betas)
        else:
            # Linear beta schedule
            betas = torch.linspace(beta_start, beta_end, timesteps)
            alphas = 1.0 - betas
            alpha_bar = torch.cumprod(alphas, dim=0)

        # SNR = alpha_bar / (1 - alpha_bar), used for min-SNR weighting
        snr = alpha_bar / (1.0 - alpha_bar).clamp(min=1e-8)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        self.register_buffer("snr", snr)

    def q_sample(self, x0, t, noise=None):
        """Forward process: add noise to clean embeddings."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]       # (B, 1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    def predict_x0_from_eps(self, xt, t, predicted_eps):
        """Recover x0 from predicted noise: x0 = (xt - sqrt(1-ab)*eps) / sqrt(ab)."""
        sqrt_ab = self.sqrt_alpha_bar[t][:, None, None]
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t][:, None, None]
        return (xt - sqrt_omab * predicted_eps) / sqrt_ab.clamp(min=1e-8)

    def _clamp_to_embeddings(self, x, embedding_weight):
        """Snap each position to its nearest valid token embedding (normalized)."""
        # x: (B, T, D), embedding_weight: (V, D)
        dist = torch.cdist(x, embedding_weight.unsqueeze(0), p=2)  # (B, T, V)
        nearest_ids = dist.argmin(dim=-1)  # (B, T)
        return embedding_weight[nearest_ids]  # (B, T, D)

    @torch.no_grad()
    def p_sample(self, model, xt, t_int, source_ids, source_mask,
                 embedding_weight=None, x0_self_cond=None):
        """Single reverse step: denoise xt -> x_{t-1}.

        Model predicts x0 (clean embeddings) directly. Optionally clamp,
        then compute the DDPM posterior.

        Returns (x_{t-1}, predicted_x0) for self-conditioning.
        """
        B = xt.shape[0]
        t = torch.full((B,), t_int, device=xt.device, dtype=torch.long)

        # Model predicts x0 directly, with self-conditioning on prev x0
        predicted_x0 = model(source_ids, source_mask, xt, t,
                             x0_self_cond=x0_self_cond)

        # Clamp: snap predicted x0 to nearest valid token embedding
        if embedding_weight is not None:
            predicted_x0 = self._clamp_to_embeddings(predicted_x0, embedding_weight)

        if t_int == 0:
            return predicted_x0, predicted_x0

        # Compute x_{t-1} from predicted x0 (DDPM posterior)
        alpha_bar_t = self.alpha_bar[t_int]
        alpha_bar_prev = self.alpha_bar[t_int - 1]
        beta_t = self.betas[t_int]

        # Posterior mean: interpolate between xt and predicted x0
        coef_x0 = beta_t * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar_t)
        coef_xt = (1.0 - alpha_bar_prev) * torch.sqrt(self.alphas[t_int]) / (1.0 - alpha_bar_t)
        mean = coef_x0 * predicted_x0 + coef_xt * xt

        # Posterior variance
        variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        noise = torch.randn_like(xt)
        x_prev = mean + torch.sqrt(variance) * noise

        return x_prev, predicted_x0

    @torch.no_grad()
    def p_sample_loop(self, model, source_ids, source_mask, seq_len, embed_dim,
                      embedding_weight=None):
        """Full reverse process: generate from pure noise with self-conditioning."""
        B = source_ids.shape[0]
        device = source_ids.device

        # Start from pure Gaussian noise
        xt = torch.randn(B, seq_len, embed_dim, device=device)
        x0_self_cond = None  # no self-cond for first step

        for t in reversed(range(self.timesteps)):
            xt, x0_pred = self.p_sample(
                model, xt, t, source_ids, source_mask,
                embedding_weight=embedding_weight,
                x0_self_cond=x0_self_cond,
            )
            # Use this step's x0 prediction as self-conditioning for next step
            x0_self_cond = x0_pred.detach()

        return xt

    @torch.no_grad()
    def ddim_sample(self, model, xt, t_int, t_prev_int, source_ids, source_mask,
                    embedding_weight=None, x0_self_cond=None):
        """Single DDIM reverse step: deterministic, no added noise."""
        B = xt.shape[0]
        t = torch.full((B,), t_int, device=xt.device, dtype=torch.long)

        predicted_x0 = model(source_ids, source_mask, xt, t,
                             x0_self_cond=x0_self_cond)

        if embedding_weight is not None:
            predicted_x0 = self._clamp_to_embeddings(predicted_x0, embedding_weight)

        if t_prev_int < 0:
            return predicted_x0, predicted_x0

        # DDIM deterministic step
        alpha_bar_t = self.alpha_bar[t_int]
        alpha_bar_prev = self.alpha_bar[t_prev_int]

        # Recover predicted noise from x0 prediction
        pred_eps = (xt - torch.sqrt(alpha_bar_t) * predicted_x0) / torch.sqrt(1 - alpha_bar_t).clamp(min=1e-8)

        # Deterministic step to t_prev
        x_prev = torch.sqrt(alpha_bar_prev) * predicted_x0 + torch.sqrt(1 - alpha_bar_prev) * pred_eps

        return x_prev, predicted_x0

    @torch.no_grad()
    def ddim_sample_loop(self, model, source_ids, source_mask, seq_len, embed_dim,
                         embedding_weight=None, ddim_steps=50):
        """DDIM reverse process with strided timesteps."""
        B = source_ids.shape[0]
        device = source_ids.device

        # Build strided timestep schedule
        step_size = self.timesteps // ddim_steps
        timesteps = list(range(self.timesteps - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        xt = torch.randn(B, seq_len, embed_dim, device=device)
        x0_self_cond = None

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            xt, x0_pred = self.ddim_sample(
                model, xt, t, t_prev, source_ids, source_mask,
                embedding_weight=embedding_weight,
                x0_self_cond=x0_self_cond,
            )
            x0_self_cond = x0_pred.detach()

        return xt

    def _ddim_denoise_from(self, model, xt, t_start, source_ids, source_mask,
                           embedding_weight=None, ddim_steps=50):
        """Run DDIM from t_start down to 0. Returns denoised x0."""
        step_size = max(1, t_start // ddim_steps)
        timesteps = list(range(t_start, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        x0_self_cond = None
        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            xt, x0_pred = self.ddim_sample(
                model, xt, t, t_prev, source_ids, source_mask,
                embedding_weight=embedding_weight,
                x0_self_cond=x0_self_cond,
            )
            x0_self_cond = x0_pred.detach()
        return xt

    @torch.no_grad()
    def ddim_progressive_sample_loop(self, model, source_ids, source_mask, seq_len, embed_dim,
                                     embedding_weight=None, ddim_steps=50,
                                     t_init=900, refinement_starts=None):
        """Truncated progressive DDIM: start from t_init (skip dead zone), then refine.

        Args:
            t_init: start from this timestep instead of T-1 (skip near-zero alpha_bar zone)
            refinement_starts: list of timesteps to restart from after first pass.
        """
        if refinement_starts is None:
            refinement_starts = [700, 500, 300]

        B = source_ids.shape[0]
        device = source_ids.device

        # Pass 1: start from noise at t_init (not t=T-1)
        xt = torch.randn(B, seq_len, embed_dim, device=device)
        x0 = self._ddim_denoise_from(model, xt, t_init, source_ids, source_mask,
                                     embedding_weight=embedding_weight, ddim_steps=ddim_steps)

        # Refinement passes: re-noise and denoise at progressively lower levels
        for t_start in refinement_starts:
            if t_start >= t_init:
                continue
            t_tensor = torch.full((B,), t_start, device=device, dtype=torch.long)
            xt, _ = self.q_sample(x0, t_tensor)
            x0 = self._ddim_denoise_from(model, xt, t_start, source_ids, source_mask,
                                         embedding_weight=embedding_weight, ddim_steps=ddim_steps)

        return x0

    @torch.no_grad()
    def iterative_refine(self, model, source_ids, source_mask, seq_len, embed_dim,
                         embedding_weight, emb_scale, tokenizer,
                         n_rounds=10, t_noise=500):
        """Source-copy initialization + iterative refinement.

        Start from noisy source embeddings (model trained with source-init),
        then denoise and iteratively refine at decreasing noise levels.
        """
        B = source_ids.shape[0]
        device = source_ids.device
        emb_norm = embedding_weight / emb_scale

        # Initialize from source embeddings + noise
        x0_src = emb_norm[source_ids]
        t_tensor = torch.full((B,), t_noise, device=device, dtype=torch.long)
        xt, _ = self.q_sample(x0_src, t_tensor)

        # Denoise with DDIM
        x0 = self._ddim_denoise_from(model, xt, t_noise, source_ids, source_mask,
                                     embedding_weight=emb_norm, ddim_steps=50)

        # Refinement passes at decreasing noise levels
        for t_refine in [300, 200, 100]:
            pred_scaled = x0 * emb_scale
            token_ids = torch.cdist(pred_scaled, embedding_weight.unsqueeze(0).float(), p=2).argmin(dim=-1)
            x0_snapped = emb_norm[token_ids]
            t_tensor = torch.full((B,), t_refine, device=device, dtype=torch.long)
            xt, _ = self.q_sample(x0_snapped, t_tensor)
            x0 = self._ddim_denoise_from(model, xt, t_refine, source_ids, source_mask,
                                         embedding_weight=emb_norm, ddim_steps=20)

        return x0

    @torch.no_grad()
    def ddim_sample_loop_infill(self, model, source_ids, source_mask,
                                known_x0, infill_mask, seq_len, embed_dim,
                                embedding_weight=None, ddim_steps=50):
        """DDIM infilling with strided timesteps."""
        B = source_ids.shape[0]
        device = source_ids.device

        step_size = self.timesteps // ddim_steps
        timesteps = list(range(self.timesteps - 1, -1, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

        xt = torch.randn(B, seq_len, embed_dim, device=device)
        infill = infill_mask.unsqueeze(-1).float()
        x0_self_cond = None

        for i, t in enumerate(timesteps):
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            denoised, x0_pred = self.ddim_sample(
                model, xt, t, t_prev, source_ids, source_mask,
                embedding_weight=embedding_weight,
                x0_self_cond=x0_self_cond,
            )
            x0_self_cond = x0_pred.detach()

            if t_prev >= 0:
                t_prev_tensor = torch.full((B,), t_prev, device=device, dtype=torch.long)
                known_noisy, _ = self.q_sample(known_x0, t_prev_tensor)
            else:
                known_noisy = known_x0

            xt = known_noisy * (1 - infill) + denoised * infill

        return xt

    @torch.no_grad()
    def p_sample_loop_infill(self, model, source_ids, source_mask,
                             known_x0, infill_mask, seq_len, embed_dim,
                             embedding_weight=None):
        """Repaint-style infilling with self-conditioning.

        Args:
            known_x0: (B, T, D) clean (normalized) embeddings for known positions
            infill_mask: (B, T) bool, True = positions to generate, False = keep known
        """
        B = source_ids.shape[0]
        device = source_ids.device

        xt = torch.randn(B, seq_len, embed_dim, device=device)
        infill = infill_mask.unsqueeze(-1).float()  # (B, T, 1)
        x0_self_cond = None

        for t in reversed(range(self.timesteps)):
            # Denoise full sequence
            denoised, x0_pred = self.p_sample(
                model, xt, t, source_ids, source_mask,
                embedding_weight=embedding_weight,
                x0_self_cond=x0_self_cond,
            )
            x0_self_cond = x0_pred.detach()

            if t > 0:
                # Re-noise known tokens to match timestep t-1
                t_prev = torch.full((B,), t - 1, device=device, dtype=torch.long)
                known_noisy, _ = self.q_sample(known_x0, t_prev)
            else:
                known_noisy = known_x0

            # Merge: known positions get the re-noised known embeddings,
            # unknown positions get the model's denoised output
            xt = known_noisy * (1 - infill) + denoised * infill

        return xt
