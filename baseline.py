"""Standard (unconstrained) transformer baseline for comparison (Experiment 1).

No block structure, no tier partitioning, no feedback — just a vanilla
GPT-style transformer matching the same d_model, n_layers, n_heads.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineTransformer(nn.Module):
    """Standard pre-norm GPT-style transformer."""

    def __init__(self, d_model, n_layers, n_heads, vocab_size,
                 max_seq_len=256, d_ff_mult=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            BaselineBlock(d_model, n_heads, d_ff_mult, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        B, T = x.shape
        tok_emb = self.embed(x)
        pos_emb = self.pos_embed(torch.arange(T, device=x.device))
        h = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            h = block(h)

        h = self.final_norm(h)
        logits = self.out_proj(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return loss, logits


class BaselineBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, d_model, n_heads, d_ff_mult, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BaselineAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * d_ff_mult),
            nn.GELU(),
            nn.Linear(d_model * d_ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BaselineAttention(nn.Module):
    """Standard multi-head causal self-attention."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_QKV = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.W_QKV(x).view(B, T, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)       # each (B, T, nh, dk)
        q = q.transpose(1, 2)              # (B, nh, T, dk)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~causal, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)
