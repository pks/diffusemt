import math
import torch
import torch.nn as nn


class SinusoidalTimestepEmbedding(nn.Module):
    """Maps scalar timestep to a vector embedding."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class DiffusionTransformer(nn.Module):
    """
    Transformer denoiser for diffusion MT.

    Takes noisy target embeddings + source tokens + timestep,
    predicts clean target embeddings (x0-prediction).
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4,
                 ff_dim=1024, dropout=0.1, max_seq_len=128):
        super().__init__()
        self.embed_dim = embed_dim

        # Source encoder
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.source_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Target (noisy embeddings + self-conditioning) projection
        # Input: noisy_target (embed_dim) concatenated with x0_self_cond (embed_dim)
        self.target_input_proj = nn.Linear(embed_dim * 2, embed_dim)

        # Target decoder (cross-attends to source)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.target_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output projection: predict clean embeddings
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def encode_source(self, source_ids, source_mask):
        """Encode source tokens."""
        B, S = source_ids.shape
        positions = torch.arange(S, device=source_ids.device).unsqueeze(0)
        x = self.token_embedding(source_ids) + self.position_embedding(positions)
        # TransformerEncoder expects src_key_padding_mask: True = ignore
        memory = self.source_encoder(x, src_key_padding_mask=~source_mask)
        return memory

    def forward(self, source_ids, source_mask, noisy_target, t, x0_self_cond=None):
        """
        Args:
            source_ids: (B, S) source token IDs
            source_mask: (B, S) bool mask (True = valid token)
            noisy_target: (B, T_len, embed_dim) noisy target embeddings
            t: (B,) timestep indices
            x0_self_cond: (B, T_len, embed_dim) previous x0 prediction for
                          self-conditioning, or None (uses zeros)
        Returns:
            predicted_x0: (B, T_len, embed_dim) predicted clean embeddings
        """
        # Encode source
        memory = self.encode_source(source_ids, source_mask)

        # Process noisy target with self-conditioning
        B, T_len, _ = noisy_target.shape
        if x0_self_cond is None:
            x0_self_cond = torch.zeros_like(noisy_target)
        target_input = torch.cat([noisy_target, x0_self_cond], dim=-1)  # (B, T, 2*D)

        positions = torch.arange(T_len, device=noisy_target.device).unsqueeze(0)
        target = self.target_input_proj(target_input) + self.position_embedding(positions)

        # Add timestep embedding (broadcast to all positions)
        t_emb = self.time_proj(self.time_embed(t))  # (B, embed_dim)
        target = target + t_emb.unsqueeze(1)

        # Decode with cross-attention to source
        decoded = self.target_decoder(
            target, memory, memory_key_padding_mask=~source_mask
        )

        # Predict clean embeddings
        return self.output_proj(decoded)
