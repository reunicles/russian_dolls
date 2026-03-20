# Matryoshka KV-Cache — Validation Prototype

## What this project is
A minimal PyTorch implementation to validate the Matryoshka KV-Cache architecture described in matryoshka_kv_cache_v2.docx in this directory. We are testing structural properties at small scale (Karpathy nanoGPT-style), not training production models.

## Architecture overview
A family of nested transformers where smaller models are literal prefixes of larger ones, sharing a single KV-cache. The weight matrix is dense with structured roles: upper-triangle preserves prefix property, lower-triangle provides feedback projections that compress higher-tier semantics into lower-tier keys.

## Two-tier test family
- **Tiny (tier 1)**: d_model=32, 8 layers, 4 heads, d_k=8, ~1.1M params
- **Small (tier 2)**: d_model=256, 16 layers, 8 heads, d_k=32, ~21M params
- Tiny is a strict architectural prefix of Small

## Key architectural components
1. **BlockNorm** — per-tier LayerNorm replacing global LayerNorm. Each tier's dims normalized independently.
2. **Tier-partitioned heads** — heads 0-3 are tier-1 (d_k=8, attend within h[:32]), heads 4-7 are tier-2 (d_k=32, attend within h[:256]). Tier-2 reads tier-1 via upper-triangular blocks.
3. **Block-upper-triangular linear layers** — W_K, W_V, W_Q, W_O, FFN all block-upper-triangular. Upper triangle = prefix property. Lower triangle = feedback projections F_ij.
4. **Low-rank feedback corrections** — delta_k ≈ c · P where c ∈ R^r (r=4) cached per token, P shared per layer/head. Full key = k_base + c · P. When standalone, c=0.
5. **Joint training** — single loss: L = λ₁·L_LM(tiny) + λ₂·L_LM(small). One forward pass, one backward pass. Not distillation — one model at multiple scales.

## Critical correctness property
The prefix must be byte-identical: running Tiny standalone must produce the same k_base values as slicing from Small's cache. This requires BlockNorm + tier-partitioned heads + block-upper-triangular W_O. If validate_cache.py fails, fix the architecture before anything else.

## Experiments to run (in order)
1. Does it train? — compare loss curves to standard baselines at d=32 and d=256
2. Is the cache byte-identical? — byte-compare Tiny standalone vs sliced from Small
3. Do feedback projections help? — ablation with F blocks zeroed vs active
4. Does joint training help Tiny? — compare Tiny in joint model vs standalone baseline

## Code structure
- model.py — MatryoshkaTransformer (the core architecture)
- baseline.py — standard unconstrained transformer for comparison
- train.py — training loop with joint Matryoshka loss
- config.py — hyperparameters
- data.py — data loading (TinyShakespeare, character-level)
- validate_cache.py — byte-identity cache test
- RESULTS.md — experiment outcomes

## Tech stack
- PyTorch (install via: pip3 install torch)
- Character-level tokenizer (no external tokenizer dependencies)
- TinyShakespeare dataset (download from karpathy's GitHub)
- Matplotlib for plots (pip3 install matplotlib)

## Conventions
- Always comment which section of the paper a component implements
- Use FP32 for the prototype (FP8 is a deployment optimization, not needed for validation)
- Keep it simple — no distributed training, no mixed precision, no fancy schedulers
- If something breaks, fix validate_cache.py (Exp 2) first — that's the foundation
