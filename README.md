# Matryoshka KV-Cache

A minimal PyTorch prototype to validate the **Matryoshka KV-Cache** architecture: a family of nested transformers where smaller models are literal prefixes of larger ones, sharing a single KV-cache.

Built at nanoGPT scale (character-level TinyShakespeare) to test structural properties, not to train production models.

## Architecture

The core idea is a dense weight matrix with structured roles:

- **Upper-triangular blocks** preserve the prefix property — running the small model standalone produces byte-identical KV-cache entries as slicing from the large model's cache.
- **Lower-triangular blocks** are feedback projections that compress higher-tier semantics into lower-tier keys via low-rank corrections.

### Two-tier test family

| Tier | d_model | Layers | Heads | d_k | Params |
|------|---------|--------|-------|-----|--------|
| Tiny | 32 | 8 | 4 | 8 | ~1.1M |
| Small | 256 | 16 | 8 | 32 | ~21M |

Tiny is a strict architectural prefix of Small. Joint training optimizes a single combined loss: `L = λ₁·L_LM(tiny) + λ₂·L_LM(small)`.

### Key components

- **BlockNorm** — per-tier LayerNorm (each tier's dimensions normalized independently)
- **Tier-partitioned heads** — tier-1 heads attend within `h[:32]`, tier-2 heads attend within `h[:256]` and read tier-1 via upper-triangular blocks
- **Block-upper-triangular linear layers** — W_K, W_V, W_Q, W_O, and FFN weights
- **Low-rank feedback corrections** — `Δk ≈ c · P` where `c ∈ R^r` (r=4) is cached per token, `P` is shared per layer/head

## Setup

```bash
pip install torch matplotlib
```

The TinyShakespeare dataset is included in `data/input.txt`.

## Usage

**Train the Matryoshka model:**
```bash
python train.py
```

**Run cache byte-identity validation:**
```bash
python validate_cache.py
```

## Project structure

```
model.py            Core MatryoshkaTransformer architecture
baseline.py         Standard unconstrained transformer for comparison
train.py            Training loop with joint Matryoshka loss
config.py           Hyperparameters (tier configs, training params)
data.py             Character-level data loading (TinyShakespeare)
validate_cache.py   Byte-identity cache test (Tiny standalone vs sliced from Small)
RESULTS.md          Experiment outcomes
```

## Experiments

1. **Does it train?** — loss curves vs standard baselines at d=32 and d=256
2. **Is the cache byte-identical?** — Tiny standalone vs sliced from Small (the foundational correctness test)
3. **Do feedback projections help?** — ablation with F blocks zeroed vs active
4. **Does joint training help Tiny?** — joint model vs standalone baseline

See [RESULTS.md](RESULTS.md) for outcomes.

## Reference

Architecture details are described in `matryoshka_kv_cache_v2.pdf`.
