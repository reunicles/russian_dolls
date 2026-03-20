"""Matryoshka KV-Cache Transformer (Sections 2-3).

A family of nested transformers sharing a single KV-cache through structural
nesting. Implements BlockNorm, tier-partitioned heads, block-upper-triangular
projections, and low-rank feedback corrections for K and V.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


# ---------------------------------------------------------------------------
# BlockNorm — per-tier LayerNorm (Section 2.4)
# ---------------------------------------------------------------------------

class BlockNorm(nn.Module):
    """Per-tier LayerNorm ensuring LN(h)[:d1] == LN(h[:d1]).

    Standard LayerNorm computes stats across ALL dims, coupling tiers.
    BlockNorm normalizes each tier's dimensions independently.
    """

    def __init__(self, tier_own_dims):
        """Args: tier_own_dims — incremental dim per tier, e.g. [32, 224]."""
        super().__init__()
        self.tier_own_dims = tier_own_dims
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in tier_own_dims])

    def forward(self, x, n_tiers=None):
        if n_tiers is None:
            n_tiers = len(self.norms)
        parts = []
        offset = 0
        for i in range(n_tiers):
            d = self.tier_own_dims[i]
            parts.append(self.norms[i](x[..., offset:offset + d]))
            offset += d
        return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# Block-upper-triangular linear layer (Section 2.2)
# ---------------------------------------------------------------------------

class BlockUpperTriangularLinear(nn.Module):
    """Linear layer with block-upper-triangular weight structure.

    y = x · W where W has blocks W^{ij} (i=input tier, j=output tier).
    Upper triangle (i <= j) is active; lower triangle is zero.
    Guarantees: output_tier_j depends only on input_tiers 0..j (prefix property).
    """

    def __init__(self, in_tier_dims, out_tier_dims):
        super().__init__()
        self.in_tier_dims = list(in_tier_dims)
        self.out_tier_dims = list(out_tier_dims)
        self.n_tiers = len(in_tier_dims)
        assert len(out_tier_dims) == self.n_tiers

        # Precompute cumulative input offsets
        self.in_offsets = [0]
        for d in self.in_tier_dims:
            self.in_offsets.append(self.in_offsets[-1] + d)

        # Create upper-triangular blocks: W^{ij} for i <= j
        self.blocks = nn.ModuleDict()
        for i in range(self.n_tiers):
            for j in range(i, self.n_tiers):
                self.blocks[f'{i}_{j}'] = nn.Linear(
                    in_tier_dims[i], out_tier_dims[j], bias=False
                )

    def forward(self, x, n_tiers=None):
        if n_tiers is None:
            n_tiers = self.n_tiers
        outputs = []
        for j in range(n_tiers):
            parts = []
            for i in range(j + 1):
                if i >= n_tiers:
                    break
                x_i = x[..., self.in_offsets[i]:self.in_offsets[i + 1]]
                parts.append(self.blocks[f'{i}_{j}'](x_i))
            outputs.append(sum(parts))
        return torch.cat(outputs, dim=-1)


# ---------------------------------------------------------------------------
# Tier-partitioned attention with low-rank feedback (Sections 2.2–2.4)
# ---------------------------------------------------------------------------

class MatryoshkaAttention(nn.Module):
    """Multi-head attention with tier-partitioned heads and feedback corrections.

    Heads 0-3 are tier-1 (d_k=8, attend within h[:32]).
    Heads 4-7 are tier-2 (d_k=32, attend within h[:256]).
    K and V for lower-tier heads receive low-rank feedback from higher tiers:
        k = k_base + c · P  where c ∈ R^r per token, P ∈ R^{r×d_k} shared.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        tiers = config.tiers
        self.n_tiers = len(tiers)
        self.feedback_rank = config.feedback_rank
        self.dropout = nn.Dropout(config.dropout)

        # Incremental dimensions per tier: [32, 224]
        self.tier_own_dims = []
        prev_d = 0
        for t in tiers:
            self.tier_own_dims.append(t.d_model - prev_d)
            prev_d = t.d_model

        # Incremental heads per tier: [4, 4]
        self.n_heads_per_tier = []
        prev_h = 0
        for t in tiers:
            self.n_heads_per_tier.append(t.n_heads - prev_h)
            prev_h = t.n_heads

        self.d_ks = [t.d_k for t in tiers]                     # [8, 32]
        self.tier_boundaries = [t.d_model for t in tiers]       # [32, 256]

        # Head output dims per tier: [4*8, 4*32] = [32, 128]
        self.head_out_dims = [
            self.n_heads_per_tier[i] * self.d_ks[i]
            for i in range(self.n_tiers)
        ]

        # Q, K, V projections — block-upper-triangular (Section 2.2)
        self.W_Q = BlockUpperTriangularLinear(self.tier_own_dims, self.head_out_dims)
        self.W_K = BlockUpperTriangularLinear(self.tier_own_dims, self.head_out_dims)
        self.W_V = BlockUpperTriangularLinear(self.tier_own_dims, self.head_out_dims)

        # Low-rank feedback for K, V (Section 2.3)
        # Only tiers with higher tiers above them receive feedback.
        self.feedback_tier_indices = []
        f_k_list, p_k_list, f_v_list, p_v_list = [], [], [], []

        for tier_idx in range(self.n_tiers):
            higher_dim = sum(self.tier_own_dims[tier_idx + 1:])
            if higher_dim > 0:
                nh = self.n_heads_per_tier[tier_idx]
                dk = self.d_ks[tier_idx]
                r = config.feedback_rank
                self.feedback_tier_indices.append(tier_idx)
                f_k_list.append(nn.Linear(higher_dim, nh * r, bias=False))
                p_k_list.append(nn.Parameter(torch.randn(nh, r, dk) * 0.01))
                f_v_list.append(nn.Linear(higher_dim, nh * r, bias=False))
                p_v_list.append(nn.Parameter(torch.randn(nh, r, dk) * 0.01))

        self.F_K_coeffs = nn.ModuleList(f_k_list)
        self.P_Ks = nn.ParameterList(p_k_list)
        self.F_V_coeffs = nn.ModuleList(f_v_list)
        self.P_Vs = nn.ParameterList(p_v_list)
        self._fb_map = {t: i for i, t in enumerate(self.feedback_tier_indices)}

        # Output projection W_O — block-upper-triangular (Section 2.4, prefix chain step 4)
        self.W_O = BlockUpperTriangularLinear(self.head_out_dims, self.tier_own_dims)

    def forward(self, x, n_tiers=None, disable_feedback=False):
        """
        Args:
            x: (B, T, d) hidden state after BlockNorm
            n_tiers: how many tiers to activate (1 = standalone Tiny)
            disable_feedback: force c=0 for cache validation
        Returns:
            output: (B, T, d) attention output
            cache_info: dict[tier_idx] -> {k_base, v_base, c_k, c_v}
        """
        if n_tiers is None:
            n_tiers = self.n_tiers
        B, T, _ = x.shape

        # Block-upper-triangular Q, K_base, V_base
        Q_all = self.W_Q(x, n_tiers=n_tiers)
        K_base_all = self.W_K(x, n_tiers=n_tiers)
        V_base_all = self.W_V(x, n_tiers=n_tiers)

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))

        head_outputs = []
        cache_info = {}
        offset = 0

        for tier_idx in range(n_tiers):
            nh = self.n_heads_per_tier[tier_idx]
            dk = self.d_ks[tier_idx]
            dim = nh * dk

            Q = Q_all[..., offset:offset + dim]
            K_base = K_base_all[..., offset:offset + dim]
            V_base = V_base_all[..., offset:offset + dim]

            # Apply low-rank feedback corrections (Section 2.3)
            K, V = K_base, V_base
            c_k_out, c_v_out = None, None

            if (not disable_feedback
                    and tier_idx in self._fb_map
                    and n_tiers > tier_idx + 1):
                fb_idx = self._fb_map[tier_idx]
                higher_start = sum(self.tier_own_dims[:tier_idx + 1])
                higher_end = sum(self.tier_own_dims[:n_tiers])
                h_higher = x[..., higher_start:higher_end]

                r = self.feedback_rank

                # Key feedback: c_k = h_higher · F_coeff, delta_k = c_k · P
                c_k = self.F_K_coeffs[fb_idx](h_higher).view(B, T, nh, r)
                delta_k = torch.einsum('bthr,hrd->bthd', c_k, self.P_Ks[fb_idx])
                K = K_base + delta_k.reshape(B, T, dim)
                c_k_out = c_k

                # Value feedback
                c_v = self.F_V_coeffs[fb_idx](h_higher).view(B, T, nh, r)
                delta_v = torch.einsum('bthr,hrd->bthd', c_v, self.P_Vs[fb_idx])
                V = V_base + delta_v.reshape(B, T, dim)
                c_v_out = c_v

            # Reshape → (B, nh, T, dk) for attention
            Q = Q.view(B, T, nh, dk).transpose(1, 2)
            K = K.view(B, T, nh, dk).transpose(1, 2)
            V = V.view(B, T, nh, dk).transpose(1, 2)

            # Scaled dot-product attention with causal mask
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dk)
            scores = scores.masked_fill(~causal[:T, :T], float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, V)
            out = out.transpose(1, 2).contiguous().view(B, T, dim)

            head_outputs.append(out)

            # Store cache info for validation
            cache_info[tier_idx] = {
                'k_base': K_base.view(B, T, nh, dk),
                'v_base': V_base.view(B, T, nh, dk),
                'c_k': c_k_out,
                'c_v': c_v_out,
            }

            offset += dim

        # Concatenate tier head outputs → W_O → residual
        output = self.W_O(torch.cat(head_outputs, dim=-1), n_tiers=n_tiers)
        return output, cache_info


# ---------------------------------------------------------------------------
# Block-upper-triangular FFN (Section 2.2)
# ---------------------------------------------------------------------------

class MatryoshkaFFN(nn.Module):
    """FFN with block-upper-triangular fc1 and fc2.

    d_ff partitioned by tier: tier_i gets d_ff_mult * tier_own_dim_i.
    GELU preserves prefix (element-wise, Section 2.4 step 5).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        tier_own_dims = []
        prev_d = 0
        for t in config.tiers:
            tier_own_dims.append(t.d_model - prev_d)
            prev_d = t.d_model

        ff_tier_dims = [d * config.d_ff_mult for d in tier_own_dims]

        self.fc1 = BlockUpperTriangularLinear(tier_own_dims, ff_tier_dims)
        self.fc2 = BlockUpperTriangularLinear(ff_tier_dims, tier_own_dims)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, n_tiers=None):
        x = self.fc1(x, n_tiers=n_tiers)
        x = F.gelu(x)
        x = self.fc2(x, n_tiers=n_tiers)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class MatryoshkaBlock(nn.Module):
    """Single pre-norm transformer layer with Matryoshka structure."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        tier_own_dims = []
        prev_d = 0
        for t in config.tiers:
            tier_own_dims.append(t.d_model - prev_d)
            prev_d = t.d_model

        self.norm1 = BlockNorm(tier_own_dims)
        self.attn = MatryoshkaAttention(config)
        self.norm2 = BlockNorm(tier_own_dims)
        self.ffn = MatryoshkaFFN(config)

    def forward(self, x, n_tiers=None, disable_feedback=False):
        # Pre-norm attention
        h = self.norm1(x, n_tiers=n_tiers)
        attn_out, cache_info = self.attn(h, n_tiers=n_tiers,
                                         disable_feedback=disable_feedback)
        x = x + attn_out

        # Pre-norm FFN
        h = self.norm2(x, n_tiers=n_tiers)
        x = x + self.ffn(h, n_tiers=n_tiers)

        return x, cache_info


# ---------------------------------------------------------------------------
# Full Matryoshka Transformer
# ---------------------------------------------------------------------------

class MatryoshkaTransformer(nn.Module):
    """Matryoshka KV-Cache Transformer with joint multi-tier training.

    Supports:
    - Joint forward: all tiers, feedback active, taps loss at each tier's
      final layer. L = sum(lambda_i * L_LM(tier_i))  (Section 3).
    - Standalone forward: single tier with c=0, equivalent to an independent
      small model (Section 6, standalone vs enriched).
    - Validation forward: full model with disable_feedback=True, for verifying
      byte-identical k_base between standalone and sliced cache (Section 2.5).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.n_tiers = len(config.tiers)

        d_model = config.tiers[-1].d_model   # 256 (full width)
        n_layers = config.tiers[-1].n_layers  # 16 (max depth)

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, d_model)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [MatryoshkaBlock(config) for _ in range(n_layers)]
        )

        # Final norm
        tier_own_dims = []
        prev_d = 0
        for t in config.tiers:
            tier_own_dims.append(t.d_model - prev_d)
            prev_d = t.d_model
        self.final_norm = BlockNorm(tier_own_dims)

        # Per-tier output heads
        self.out_proj = nn.ModuleList([
            nn.Linear(t.d_model, config.vocab_size, bias=False)
            for t in config.tiers
        ])

        # Initialize weights
        self.apply(self._init_weights)
        self._init_feedback_small()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_feedback_small(self):
        """Re-initialize feedback projections with small weights (Section 2.3).

        Ensures the model starts near standalone behavior (c ≈ 0).
        """
        for block in self.blocks:
            for fc in block.attn.F_K_coeffs:
                nn.init.normal_(fc.weight, std=0.001)
            for fc in block.attn.F_V_coeffs:
                nn.init.normal_(fc.weight, std=0.001)

    def forward(self, x, targets=None, n_tiers=None, disable_feedback=False):
        """
        Args:
            x: (B, T) token indices
            targets: (B, T) target token indices (optional)
            n_tiers: number of tiers to use (None = all)
            disable_feedback: force c=0 for cache validation
        Returns:
            losses: dict[tier_idx -> cross-entropy loss]
            logits: dict[tier_idx -> (B, T, vocab) logits]
            all_cache: list of per-layer cache_info dicts
        """
        if n_tiers is None:
            n_tiers = self.n_tiers

        B, T = x.shape
        d_active = self.config.tiers[n_tiers - 1].d_model
        n_layers = self.config.tiers[n_tiers - 1].n_layers

        # Token + positional embeddings, sliced to active width
        tok_emb = self.embed(x)[..., :d_active]
        pos = torch.arange(T, device=x.device)
        pos_emb = self.pos_embed(pos)[:, :d_active]
        h = self.drop(tok_emb + pos_emb)

        all_cache = []
        losses = {}
        logits = {}

        for layer_idx in range(n_layers):
            h, cache_info = self.blocks[layer_idx](
                h, n_tiers=n_tiers, disable_feedback=disable_feedback
            )
            all_cache.append(cache_info)

            # Tap output at each tier's final layer for its loss
            for tier_idx in range(n_tiers):
                tier = self.config.tiers[tier_idx]
                if layer_idx == tier.n_layers - 1:
                    d_tier = tier.d_model
                    h_tier = h[..., :d_tier]
                    h_norm = self.final_norm(h_tier, n_tiers=tier_idx + 1)
                    tier_logits = self.out_proj[tier_idx](h_norm)
                    logits[tier_idx] = tier_logits

                    if targets is not None:
                        loss = F.cross_entropy(
                            tier_logits.view(-1, self.config.vocab_size),
                            targets.view(-1)
                        )
                        losses[tier_idx] = loss

        return losses, logits, all_cache

    def count_parameters(self, tier=None):
        """Count trainable parameters, optionally for a specific tier."""
        if tier is None:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Rough estimate — for full model only
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
