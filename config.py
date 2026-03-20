"""Hyperparameters for the Matryoshka KV-Cache prototype (Section 4, Table 1)."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class TierConfig:
    """Configuration for a single tier in the nested model family."""
    d_model: int      # total model dimension up to this tier
    n_layers: int     # number of layers this tier uses
    n_heads: int      # total number of heads up to this tier
    d_k: int          # key/value dimension per head = d_model / n_heads


# Two-tier test family (Table 1)
TINY = TierConfig(d_model=32, n_layers=8, n_heads=4, d_k=8)
SMALL = TierConfig(d_model=256, n_layers=16, n_heads=8, d_k=32)


@dataclass
class ModelConfig:
    """Full Matryoshka model configuration."""
    tiers: List[TierConfig] = field(default_factory=lambda: [TINY, SMALL])
    feedback_rank: int = 4        # r in low-rank correction delta_k ~ c * P (Section 2.3)
    d_ff_mult: int = 4            # FFN hidden dim multiplier
    max_seq_len: int = 256        # maximum sequence length
    vocab_size: int = 65          # set from data (TinyShakespeare char-level)
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """Training hyperparameters (Section 3)."""
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_iters: int = 5000
    warmup_iters: int = 100
    eval_interval: int = 500
    eval_iters: int = 200
    lambdas: List[float] = field(default_factory=lambda: [1.0, 1.0])  # lambda_1, lambda_2
    device: str = 'cpu'           # auto-detected in train.py
