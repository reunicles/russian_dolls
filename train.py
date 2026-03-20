"""Training loop with joint Matryoshka loss (Section 3).

Supports training:
- Matryoshka model (joint multi-tier loss)
- Matryoshka model with feedback disabled (Experiment 3 ablation)
- Baseline transformers (for comparison in Experiment 1)

Usage:
    python3 train.py                              # joint Matryoshka training
    python3 train.py --disable-feedback           # ablation: no feedback (Exp 3)
    python3 train.py --baseline tiny              # baseline d=32
    python3 train.py --baseline small             # baseline d=256
"""

import argparse
import json
import time
import math
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import ModelConfig, TrainConfig, TINY, SMALL
from data import ShakespeareDataset
from model import MatryoshkaTransformer
from baseline import BaselineTransformer


def get_device():
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters, batch_size, device,
                  mode='matryoshka', disable_feedback=False):
    """Estimate train and val loss over eval_iters batches."""
    model.eval()
    results = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, device)
            if mode == 'matryoshka':
                tier_losses, _, _ = model(x, targets=y,
                                          disable_feedback=disable_feedback)
                # Report per-tier losses
                for tier_idx, loss in tier_losses.items():
                    key = f'{split}_tier{tier_idx}'
                    if key not in results:
                        results[key] = []
                    results[key].append(loss.item())
            else:
                loss, _ = model(x, targets=y)
                key = f'{split}'
                if key not in results:
                    results[key] = []
                results[key].append(loss.item())
        losses.clear()
    # Average
    return {k: sum(v) / len(v) for k, v in results.items()}


def get_lr(it, warmup_iters, max_iters, max_lr, min_lr_ratio=0.1):
    """Cosine learning rate schedule with linear warmup."""
    min_lr = max_lr * min_lr_ratio
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_matryoshka(train_cfg, model_cfg, dataset, device,
                     disable_feedback=False, tag='matryoshka'):
    """Joint Matryoshka training (Section 3).

    Args:
        disable_feedback: if True, train with c=0 (Experiment 3 ablation).
        tag: prefix for saved files.
    """
    model_cfg.vocab_size = dataset.vocab_size
    model = MatryoshkaTransformer(model_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fb_str = ' (feedback DISABLED)' if disable_feedback else ''
    print(f'Matryoshka model{fb_str}: {n_params:,} parameters')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    history = {'iter': [], 'train_loss': [], 'val_loss': []}
    tier_history = {i: {'train': [], 'val': []} for i in range(len(model_cfg.tiers))}

    t0 = time.time()
    for it in range(train_cfg.max_iters):
        # Learning rate schedule
        lr = get_lr(it, train_cfg.warmup_iters, train_cfg.max_iters,
                    train_cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Evaluate periodically
        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters - 1:
            metrics = estimate_loss(model, dataset, train_cfg.eval_iters,
                                    train_cfg.batch_size, device,
                                    mode='matryoshka',
                                    disable_feedback=disable_feedback)
            dt = time.time() - t0
            print(f'iter {it:5d} | time {dt:.1f}s | lr {lr:.2e}', end='')
            for k, v in sorted(metrics.items()):
                print(f' | {k} {v:.4f}', end='')
            print()

            history['iter'].append(it)
            for tier_idx in range(len(model_cfg.tiers)):
                train_key = f'train_tier{tier_idx}'
                val_key = f'val_tier{tier_idx}'
                if train_key in metrics:
                    tier_history[tier_idx]['train'].append(metrics[train_key])
                    tier_history[tier_idx]['val'].append(metrics[val_key])

        # Training step
        model.train()
        x, y = dataset.get_batch('train', train_cfg.batch_size, device)
        tier_losses, _, _ = model(x, targets=y,
                                  disable_feedback=disable_feedback)

        # Joint loss: L = sum(lambda_i * L_LM(tier_i))  (Section 3)
        total_loss = sum(
            train_cfg.lambdas[i] * tier_losses[i]
            for i in tier_losses
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Save model and history
    torch.save(model.state_dict(), f'{tag}_model.pt')
    hist_data = {'iter': history['iter'], 'tier_history': {
        str(k): v for k, v in tier_history.items()
    }}
    with open(f'{tag}_history.json', 'w') as f:
        json.dump(hist_data, f)
    print(f'\nModel saved to {tag}_model.pt')

    # Plot loss curves
    plot_matryoshka_losses(history['iter'], tier_history, model_cfg, tag)

    return model, tier_history


def train_baseline(name, train_cfg, dataset, device):
    """Train a standard baseline transformer."""
    if name == 'tiny':
        d_model, n_layers, n_heads = 32, 8, 4
    elif name == 'small':
        d_model, n_layers, n_heads = 256, 16, 8
    else:
        raise ValueError(f'Unknown baseline: {name}')

    model = BaselineTransformer(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        vocab_size=dataset.vocab_size, max_seq_len=dataset.block_size,
        dropout=0.1,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Baseline {name} (d={d_model}, L={n_layers}): {n_params:,} parameters')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )

    history = {'iter': [], 'train': [], 'val': []}

    t0 = time.time()
    for it in range(train_cfg.max_iters):
        lr = get_lr(it, train_cfg.warmup_iters, train_cfg.max_iters,
                    train_cfg.learning_rate)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters - 1:
            metrics = estimate_loss(model, dataset, train_cfg.eval_iters,
                                    train_cfg.batch_size, device, mode='baseline')
            dt = time.time() - t0
            print(f'iter {it:5d} | time {dt:.1f}s | lr {lr:.2e}'
                  f' | train {metrics["train"]:.4f} | val {metrics["val"]:.4f}')
            history['iter'].append(it)
            history['train'].append(metrics['train'])
            history['val'].append(metrics['val'])

        model.train()
        x, y = dataset.get_batch('train', train_cfg.batch_size, device)
        loss, _ = model(x, targets=y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    torch.save(model.state_dict(), f'baseline_{name}.pt')
    with open(f'baseline_{name}_history.json', 'w') as f:
        json.dump(history, f)
    print(f'\nModel saved to baseline_{name}.pt')

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(history['iter'], history['train'], label='train')
    plt.plot(history['iter'], history['val'], label='val')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title(f'Baseline {name} (d={d_model}) Loss Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'baseline_{name}_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Loss plot saved to baseline_{name}_loss.png')

    return model, history


def plot_matryoshka_losses(iters, tier_history, config, tag='matryoshka'):
    """Plot per-tier loss curves for the Matryoshka model."""
    fig, axes = plt.subplots(1, len(config.tiers), figsize=(6 * len(config.tiers), 5))
    if len(config.tiers) == 1:
        axes = [axes]

    tier_names = ['Tiny (tier 1)', 'Small (tier 2)', 'Medium (tier 3)',
                  'Large (tier 4)', 'Huge (tier 5)']

    for tier_idx in range(len(config.tiers)):
        ax = axes[tier_idx]
        if tier_history[tier_idx]['train']:
            ax.plot(iters[:len(tier_history[tier_idx]['train'])],
                    tier_history[tier_idx]['train'], label='train')
            ax.plot(iters[:len(tier_history[tier_idx]['val'])],
                    tier_history[tier_idx]['val'], label='val')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        name = tier_names[tier_idx] if tier_idx < len(tier_names) else f'Tier {tier_idx+1}'
        t = config.tiers[tier_idx]
        ax.set_title(f'{name} (d={t.d_model}, L={t.n_layers})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{tag}_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Loss plot saved to {tag}_loss.png')


def main():
    parser = argparse.ArgumentParser(description='Matryoshka KV-Cache Training')
    parser.add_argument('--baseline', type=str, choices=['tiny', 'small'],
                        help='Train a baseline instead of Matryoshka')
    parser.add_argument('--disable-feedback', action='store_true',
                        help='Ablation: train with c=0, no feedback (Exp 3)')
    parser.add_argument('--max-iters', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-interval', type=int, default=500)
    args = parser.parse_args()

    device = get_device()
    print(f'Using device: {device}')

    train_cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
    )

    dataset = ShakespeareDataset(block_size=args.seq_len)

    if args.baseline:
        train_baseline(args.baseline, train_cfg, dataset, device)
    else:
        model_cfg = ModelConfig(vocab_size=dataset.vocab_size,
                                max_seq_len=args.seq_len)
        tag = 'matryoshka_no_feedback' if args.disable_feedback else 'matryoshka'
        train_matryoshka(train_cfg, model_cfg, dataset, device,
                         disable_feedback=args.disable_feedback, tag=tag)


if __name__ == '__main__':
    main()
