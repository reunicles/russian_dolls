"""Byte-identity cache test (Experiment 2, Section 2.5).

Verifies the critical correctness property: running Tiny standalone produces
the same k_base values as slicing tier-1 keys from the full model's cache
(with feedback disabled, c=0).

This validates that BlockNorm + tier-partitioned heads + block-upper-triangular
W_O correctly preserve the prefix through all layers.
"""

import torch
from config import ModelConfig
from model import MatryoshkaTransformer


def validate_cache(config=None, verbose=True):
    """Run the byte-identity cache validation.

    Returns True if all k_base values match, False otherwise.
    """
    if config is None:
        config = ModelConfig()

    device = 'cpu'  # FP32 on CPU for exact comparison
    model = MatryoshkaTransformer(config).to(device)
    model.eval()

    # Random input
    torch.manual_seed(42)
    B, T = 2, 32
    x = torch.randint(0, config.vocab_size, (B, T), device=device)

    with torch.no_grad():
        # 1) Standalone Tiny: n_tiers=1 (only tier-1, 8 layers, 32-dim)
        _, _, cache_standalone = model(x, n_tiers=1)

        # 2) Full model with feedback disabled (c=0)
        _, _, cache_full = model(x, n_tiers=2, disable_feedback=True)

    # Compare k_base for tier-1 heads across shared layers (0-7)
    n_shared_layers = config.tiers[0].n_layers
    all_match = True

    for layer_idx in range(n_shared_layers):
        k_base_standalone = cache_standalone[layer_idx][0]['k_base']
        k_base_full = cache_full[layer_idx][0]['k_base']

        v_base_standalone = cache_standalone[layer_idx][0]['v_base']
        v_base_full = cache_full[layer_idx][0]['v_base']

        k_match = torch.equal(k_base_standalone, k_base_full)
        v_match = torch.equal(v_base_standalone, v_base_full)

        if verbose:
            if k_match and v_match:
                print(f'  Layer {layer_idx:2d}: k_base MATCH, v_base MATCH')
            else:
                k_diff = (k_base_standalone - k_base_full).abs().max().item()
                v_diff = (v_base_standalone - v_base_full).abs().max().item()
                print(f'  Layer {layer_idx:2d}: k_base {"MATCH" if k_match else f"FAIL (max diff {k_diff:.2e})"}, '
                      f'v_base {"MATCH" if v_match else f"FAIL (max diff {v_diff:.2e})"}')

        if not k_match or not v_match:
            all_match = False

    return all_match


def main():
    print('=' * 60)
    print('Experiment 2: Cache Byte-Identity Validation')
    print('=' * 60)
    print()
    print('Comparing k_base between standalone Tiny (n_tiers=1)')
    print('and full model with c=0 (n_tiers=2, disable_feedback=True)')
    print()

    passed = validate_cache()

    print()
    if passed:
        print('PASSED: All k_base and v_base values are byte-identical.')
        print('The prefix property holds — BlockNorm + tier-partitioned heads')
        print('+ block-upper-triangular W_O correctly preserve tier-1 cache.')
    else:
        print('FAILED: k_base values differ between standalone and sliced cache.')
        print('The prefix property is broken — fix the architecture before proceeding.')

    return passed


if __name__ == '__main__':
    main()
