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


class PreLNEncoderLayer(nn.Module):
    """Pre-LayerNorm transformer encoder layer (more stable for from-scratch training)."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, src_key_padding_mask=None, attn_mask=None):
        # Pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=src_key_padding_mask,
                              attn_mask=attn_mask)
        x = x + h
        # Pre-norm feedforward
        x = x + self.ff(self.norm2(x))
        return x


class PreLNDecoderLayer(nn.Module):
    """Pre-LayerNorm transformer decoder layer with cross-attention."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                attn_mask=None):
        # Pre-norm self-attention (with optional causal mask)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h, key_padding_mask=tgt_key_padding_mask,
                              attn_mask=attn_mask)
        x = x + h
        # Pre-norm cross-attention
        h = self.norm2(x)
        h, _ = self.cross_attn(h, memory, memory, key_padding_mask=memory_key_padding_mask)
        x = x + h
        # Pre-norm feedforward
        x = x + self.ff(self.norm3(x))
        return x


class SourceCorruptionEncoderDecoder(nn.Module):
    """From-scratch encoder-decoder for source-as-corruption discrete diffusion.

    Only the word embeddings are pretrained (frozen mBERT). All transformer layers,
    position embeddings, timestep conditioning, and output head are trained from scratch.

    Encoder: processes English source (never corrupted)
    Decoder: processes corrupted target with cross-attention to encoder
    """

    def __init__(self, pretrained_name="bert-base-multilingual-cased",
                 model_dim=512, embed_dim=768, num_heads=8,
                 encoder_layers=6, decoder_layers=6, ff_dim=2048,
                 dropout=0.1, max_seq_len=128, freeze_embeddings=True):
        super().__init__()
        from transformers import BertModel

        bert = BertModel.from_pretrained(pretrained_name)
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.vocab_size = bert.config.vocab_size

        # Frozen pretrained word embeddings (input)
        self.word_embeddings = bert.embeddings.word_embeddings
        if freeze_embeddings:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        # Frozen output embedding (initialized from pretrained)
        # Using frozen embedding as output anchor prevents collapse
        self.register_buffer("output_embedding",
                             bert.embeddings.word_embeddings.weight.data.clone())

        del bert

        # Project from embedding dim to model dim
        self.embed_proj = nn.Linear(embed_dim, model_dim)

        # Learned position embeddings (from scratch)
        self.pos_embedding = nn.Embedding(max_seq_len, model_dim)
        self.embed_norm = nn.LayerNorm(model_dim)
        self.embed_drop = nn.Dropout(dropout)

        # Timestep conditioning (decoder only)
        self.time_embed = SinusoidalTimestepEmbedding(model_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # From-scratch encoder (processes source English)
        self.encoder_layers = nn.ModuleList([
            PreLNEncoderLayer(model_dim, num_heads, ff_dim, dropout)
            for _ in range(encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(model_dim)

        # From-scratch decoder (processes corrupted target with cross-attn to source)
        self.decoder_layers = nn.ModuleList([
            PreLNDecoderLayer(model_dim, num_heads, ff_dim, dropout)
            for _ in range(decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(model_dim)

        # Output head: project model_dim -> embed_dim, then tied projection to vocab
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, embed_dim),
            nn.GELU(),
        )
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

        # Auxiliary encoder MLM head (small, for anti-collapse gradient signal)
        self.aux_mlm_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        """Careful initialization for from-scratch training stability."""
        n_layers = len(self.encoder_layers) + len(self.decoder_layers)

        for module in [self.embed_proj, self.time_proj, self.aux_mlm_head]:
            for m in module.modules() if hasattr(module, 'modules') else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Encoder/decoder layers: scale residual contributions by 1/sqrt(2*n_layers)
        scale = 1.0 / math.sqrt(2.0 * n_layers)
        for layer in list(self.encoder_layers) + list(self.decoder_layers):
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            # Scale the output projections of attention and FF
            if hasattr(layer, 'self_attn'):
                layer.self_attn.out_proj.weight.data *= scale
            if hasattr(layer, 'cross_attn'):
                layer.cross_attn.out_proj.weight.data *= scale
            # Scale last FF linear
            ff_layers = [m for m in layer.ff.modules() if isinstance(m, nn.Linear)]
            if ff_layers:
                ff_layers[-1].weight.data *= scale

        # Output projection: small init so initial logits are low-magnitude
        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(n_layers))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Position embeddings
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def encode_source(self, source_ids, source_mask):
        """Encode English source tokens."""
        B, L = source_ids.shape
        pos = torch.arange(L, device=source_ids.device).unsqueeze(0)

        x = self.embed_proj(self.word_embeddings(source_ids)) + self.pos_embedding(pos)
        x = self.embed_drop(self.embed_norm(x))

        # key_padding_mask: True = IGNORE (PyTorch convention)
        pad_mask = ~source_mask

        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=pad_mask)

        return self.encoder_norm(x)

    def forward(self, input_ids, padding_mask, t,
                source_ids=None, source_mask=None, encoder_output=None,
                causal=False):
        """
        Args:
            input_ids: (B, L) corrupted target tokens (or shifted target for AR)
            padding_mask: (B, L) bool, True = valid target position
            t: (B,) timestep indices
            source_ids: (B, L) English source tokens (needed if encoder_output is None)
            source_mask: (B, L) bool, True = valid source position
            encoder_output: (B, L, model_dim) precomputed encoder output (optional)
            causal: bool, if True use causal self-attention mask (for AR phase)
        Returns:
            logits: (B, L, vocab_size) predictions for target tokens
        """
        B, L = input_ids.shape

        # Encode source if not provided
        if encoder_output is None:
            assert source_ids is not None and source_mask is not None
            encoder_output = self.encode_source(source_ids, source_mask)
            src_pad_mask = ~source_mask
        else:
            src_pad_mask = ~source_mask if source_mask is not None else None

        # Decoder input: corrupted target
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed_proj(self.word_embeddings(input_ids)) + self.pos_embedding(pos)
        x = self.embed_drop(self.embed_norm(x))

        # Add timestep conditioning
        t_emb = self.time_proj(self.time_embed(t))
        x = x + t_emb.unsqueeze(1)

        tgt_pad_mask = ~padding_mask

        # Causal mask for autoregressive phase
        attn_mask = None
        if causal:
            attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                L, device=input_ids.device)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output,
                      tgt_key_padding_mask=tgt_pad_mask,
                      memory_key_padding_mask=src_pad_mask,
                      attn_mask=attn_mask)

        x = self.decoder_norm(x)

        # Output projection via frozen output embedding
        h = self.output_proj(x)
        logits = h @ self.output_embedding.T + self.output_bias
        return logits

    def aux_mlm_logits(self, encoder_output):
        """Compute auxiliary MLM logits from encoder output."""
        h = self.aux_mlm_head(encoder_output)
        return h @ self.output_embedding.T


class SourceCorruptionEncoderOnly(nn.Module):
    """Deep encoder-only model for source-as-corruption discrete diffusion.

    Concatenates [source | corrupted_target] into a single sequence with
    segment embeddings, processes everything with a deep Pre-LN transformer.
    Predicts target tokens from target positions.

    More parameters go to one powerful encoder instead of split enc/dec.
    """

    def __init__(self, pretrained_name="bert-base-multilingual-cased",
                 model_dim=512, embed_dim=768, num_heads=8,
                 num_layers=12, ff_dim=2048,
                 dropout=0.1, max_seq_len=128, freeze_embeddings=True):
        super().__init__()
        from transformers import BertModel

        bert = BertModel.from_pretrained(pretrained_name)
        self.embed_dim = embed_dim
        self.model_dim = model_dim
        self.vocab_size = bert.config.vocab_size
        self.max_seq_len = max_seq_len

        # Frozen pretrained word embeddings
        self.word_embeddings = bert.embeddings.word_embeddings
        if freeze_embeddings:
            for p in self.word_embeddings.parameters():
                p.requires_grad = False

        # Frozen output embedding
        self.register_buffer("output_embedding",
                             bert.embeddings.word_embeddings.weight.data.clone())

        del bert

        # Project from embedding dim to model dim
        self.embed_proj = nn.Linear(embed_dim, model_dim)

        # Learned position embeddings (for concatenated sequence, 2x max_seq_len)
        self.pos_embedding = nn.Embedding(max_seq_len * 2, model_dim)

        # Segment embeddings (0=source, 1=target)
        self.segment_embedding = nn.Embedding(2, model_dim)

        self.embed_norm = nn.LayerNorm(model_dim)
        self.embed_drop = nn.Dropout(dropout)

        # Timestep conditioning
        self.time_embed = SinusoidalTimestepEmbedding(model_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

        # Deep encoder
        self.layers = nn.ModuleList([
            PreLNEncoderLayer(model_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.output_norm = nn.LayerNorm(model_dim)

        # Output head
        self.output_proj = nn.Sequential(
            nn.Linear(model_dim, embed_dim),
            nn.GELU(),
        )
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

        # Auxiliary MLM head (for encoder gradient signal)
        self.aux_mlm_head = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, embed_dim),
        )

        self._init_weights()

    def _init_weights(self):
        n_layers = len(self.layers)
        for module in [self.embed_proj, self.time_proj, self.aux_mlm_head]:
            for m in module.modules() if hasattr(module, 'modules') else [module]:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        scale = 1.0 / math.sqrt(2.0 * n_layers)
        for layer in self.layers:
            for m in layer.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            if hasattr(layer, 'self_attn'):
                layer.self_attn.out_proj.weight.data *= scale
            ff_layers = [m for m in layer.ff.modules() if isinstance(m, nn.Linear)]
            if ff_layers:
                ff_layers[-1].weight.data *= scale

        for m in self.output_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02 / math.sqrt(n_layers))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        nn.init.normal_(self.segment_embedding.weight, std=0.02)

    def encode_source(self, source_ids, source_mask):
        """For compatibility with diffusion code that pre-computes encoder output.
        Returns None — this model doesn't separate encoding."""
        return None

    def forward(self, input_ids, padding_mask, t,
                source_ids=None, source_mask=None, encoder_output=None,
                causal=False):
        """
        Args:
            input_ids: (B, L) corrupted target tokens
            padding_mask: (B, L) bool, True = valid target position
            t: (B,) timestep indices
            source_ids: (B, L_src) English source tokens
            source_mask: (B, L_src) bool, True = valid source position
        Returns:
            logits: (B, L, vocab_size) predictions for target tokens
        """
        B, L_tgt = input_ids.shape
        L_src = source_ids.shape[1] if source_ids is not None else 0
        device = input_ids.device

        # Concatenate [source | target]
        concat_ids = torch.cat([source_ids, input_ids], dim=1)  # (B, L_src + L_tgt)
        L_total = concat_ids.shape[1]

        # Position embeddings
        pos = torch.arange(L_total, device=device).unsqueeze(0)
        x = self.embed_proj(self.word_embeddings(concat_ids)) + self.pos_embedding(pos)

        # Segment embeddings
        seg = torch.cat([
            torch.zeros(B, L_src, device=device, dtype=torch.long),
            torch.ones(B, L_tgt, device=device, dtype=torch.long),
        ], dim=1)
        x = x + self.segment_embedding(seg)

        x = self.embed_drop(self.embed_norm(x))

        # Add timestep conditioning (to target positions only, or all)
        t_emb = self.time_proj(self.time_embed(t))
        # Add to all positions (model learns to use it for target denoising)
        x = x + t_emb.unsqueeze(1)

        # Padding mask: True=IGNORE for PyTorch convention
        concat_pad = torch.cat([~source_mask, ~padding_mask], dim=1)

        # Causal mask for AR phase
        attn_mask = None
        if causal:
            # Causal mask only on the target portion — source positions visible to all
            attn_mask = torch.zeros(L_total, L_total, device=device)
            # Target positions can't attend to future target positions
            tgt_causal = torch.nn.Transformer.generate_square_subsequent_mask(
                L_tgt, device=device)
            attn_mask[L_src:, L_src:] = tgt_causal

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=concat_pad, attn_mask=attn_mask)

        x = self.output_norm(x)

        # Extract target positions and project to vocab
        target_hidden = x[:, L_src:, :]  # (B, L_tgt, model_dim)
        h = self.output_proj(target_hidden)
        logits = h @ self.output_embedding.T + self.output_bias
        return logits

    def aux_mlm_logits_from_concat(self, x, L_src):
        """Get MLM logits for source positions from concatenated output."""
        source_hidden = x[:, :L_src, :]
        h = self.aux_mlm_head(source_hidden)
        return h @ self.output_embedding.T


class LengthPredictor(nn.Module):
    """Standalone target length predictor from source embeddings.

    Separate from the main model to avoid DDP conflicts.
    Shares frozen mBERT embeddings with the main model to save GPU memory.
    """

    def __init__(self, word_embeddings, embed_dim=768, hidden_dim=512):
        """
        Args:
            word_embeddings: nn.Embedding — shared frozen mBERT embeddings (not owned)
        """
        super().__init__()
        self.word_embeddings = word_embeddings  # shared reference, not a copy

        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, source_ids, source_mask):
        """Predict target token count from source.

        Args:
            source_ids: (B, L_src) source token IDs
            source_mask: (B, L_src) bool, True = real token
        Returns:
            (B,) predicted target length
        """
        with torch.no_grad():
            x = self.word_embeddings(source_ids)  # (B, L_src, embed_dim)
        x = self.proj(x)  # (B, L_src, hidden_dim)
        mask = source_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)
