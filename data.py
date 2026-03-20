"""Data loading for TinyShakespeare, character-level (Section 4)."""

import os
import urllib.request
import torch


DATA_URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'


def download_shakespeare(data_dir='data'):
    """Download TinyShakespeare dataset if not present."""
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, 'input.txt')
    if not os.path.exists(filepath):
        print('Downloading TinyShakespeare...')
        urllib.request.urlretrieve(DATA_URL, filepath)
        print('Done.')
    with open(filepath, 'r') as f:
        text = f.read()
    return text


class CharTokenizer:
    """Character-level tokenizer."""

    def __init__(self, text):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for i, c in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids)


class ShakespeareDataset:
    """Character-level Shakespeare dataset with train/val split."""

    def __init__(self, block_size=256, data_dir='data', val_frac=0.1):
        text = download_shakespeare(data_dir)
        self.tokenizer = CharTokenizer(text)
        self.vocab_size = self.tokenizer.vocab_size
        self.block_size = block_size

        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        split = int(len(data) * (1 - val_frac))
        self.train_data = data[:split]
        self.val_data = data[split:]
        print(f'Vocab size: {self.vocab_size}, '
              f'Train tokens: {len(self.train_data):,}, '
              f'Val tokens: {len(self.val_data):,}')

    def get_batch(self, split, batch_size, device='cpu'):
        """Get a random batch of (input, target) pairs."""
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)
