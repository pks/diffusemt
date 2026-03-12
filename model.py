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
    Encoder-only transformer for discrete diffusion MT.

    Input: [source tokens | corrupted target tokens] as a single sequence.
    Source tokens are never masked. Target tokens are masked by the diffusion process.
    Segment embeddings distinguish source (0) from target (1).
    Output: logits over vocabulary for each position.
    """

    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4,
                 ff_dim=1024, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        # Token + position + segment embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.segment_embedding = nn.Embedding(2, embed_dim)  # 0=source, 1=target

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(embed_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Bidirectional transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output head: predict token logits
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size, bias=False)

        # Tie output projection weights to token embedding
        self.output_proj.weight = self.token_embedding.weight

    def forward(self, input_ids, padding_mask, segment_ids, t):
        """
        Args:
            input_ids: (B, S+T) concatenated [source | corrupted_target] token IDs
            padding_mask: (B, S+T) bool mask (True = valid, False = padding)
            segment_ids: (B, S+T) 0 for source positions, 1 for target positions
            t: (B,) timestep indices
        Returns:
            logits: (B, S+T, vocab_size) predicted token logits
        """
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = (self.token_embedding(input_ids)
             + self.position_embedding(positions)
             + self.segment_embedding(segment_ids))

        # Add timestep embedding (broadcast to all positions)
        t_emb = self.time_proj(self.time_embed(t))  # (B, embed_dim)
        x = x + t_emb.unsqueeze(1)

        # Bidirectional self-attention over full sequence
        x = self.encoder(x, src_key_padding_mask=~padding_mask)

        # Predict token logits (scale for tied embeddings)
        return self.output_proj(self.output_norm(x)) / (self.embed_dim ** 0.5)
