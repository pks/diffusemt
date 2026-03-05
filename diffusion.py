import torch
import torch.nn as nn


class GaussianDiffusion(nn.Module):
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.timesteps = timesteps

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

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
