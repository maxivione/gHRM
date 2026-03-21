"""Train CausalHRM on TinyStories.

Usage:
    python scripts/train_tinystories.py [--hidden_dim 512] [--max_steps 10000]

Downloads TinyStories automatically on first run.
"""
import os
import sys
import time
import math
import json
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.causal_hrm import CausalHRM, CausalHRMConfig


def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument('--hidden_dim', type=int, default=384)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--num_h_layers', type=int, default=3)
    p.add_argument('--num_l_layers', type=int, default=3)
    p.add_argument('--max_seq_len', type=int, default=512)
    p.add_argument('--max_act_steps', type=int, default=4)
    # Training
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--max_steps', type=int, default=10000)
    p.add_argument('--warmup_steps', type=int, default=500)
    p.add_argument('--weight_decay', type=float, default=0.1)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--log_interval', type=int, default=50)
    p.add_argument('--save_interval', type=int, default=2000)
    p.add_argument('--eval_interval', type=int, default=500)
    # Data
    p.add_argument('--data_dir', type=str, default='data/tinystories')
    p.add_argument('--checkpoint_dir', type=str, default='checkpoints/causal_hrm_tinystories')
    p.add_argument('--compile', action='store_true')
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float32', 'bfloat16', 'float16'])
    return p.parse_args()


class TinyStoriesDataset(Dataset):
    """Tokenized TinyStories dataset, stored as memory-mapped .bin file."""
    def __init__(self, data_path: str, seq_len: int):
        import numpy as np
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start:start + self.seq_len + 1].astype(int)
        x = torch.from_numpy(chunk[:-1]).long()
        y = torch.from_numpy(chunk[1:]).long()
        return x, y


def prepare_tinystories(data_dir: str):
    """Download and tokenize TinyStories if not already done."""
    import numpy as np

    train_bin = os.path.join(data_dir, 'train.bin')
    val_bin = os.path.join(data_dir, 'val.bin')

    if os.path.exists(train_bin) and os.path.exists(val_bin):
        return train_bin, val_bin

    os.makedirs(data_dir, exist_ok=True)
    print('Downloading TinyStories...')

    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    for split, out_path in [('train', train_bin), ('validation', val_bin)]:
        print(f'Tokenizing {split}...')
        all_tokens = []
        for example in ds[split]:
            text = example['text']
            tokens = enc.encode(text, allowed_special="all")
            all_tokens.extend(tokens + [eot])

        arr = np.array(all_tokens, dtype=np.uint16)
        arr.tofile(out_path)
        print(f'  {split}: {len(arr):,} tokens -> {out_path}')

    return train_bin, val_bin


def get_lr(step: int, warmup: int, max_steps: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, max_steps - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(model, val_loader, max_batches=20, device='cuda', dtype=torch.bfloat16):
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.autocast('cuda', dtype=dtype):
            out = model(x, labels=y)
        total_loss += out['loss'].item()
        count += 1
    model.train()
    return total_loss / max(count, 1)


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype_map = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}
    amp_dtype = dtype_map[args.dtype]

    torch.set_float32_matmul_precision('high')

    # Data
    train_bin, val_bin = prepare_tinystories(args.data_dir)
    train_ds = TinyStoriesDataset(train_bin, args.max_seq_len)
    val_ds = TinyStoriesDataset(val_bin, args.max_seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=1, pin_memory=True, drop_last=True)

    # Model
    config = CausalHRMConfig(
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_h_layers=args.num_h_layers,
        num_l_layers=args.num_l_layers,
        max_seq_len=args.max_seq_len,
        max_act_steps=args.max_act_steps,
    )
    model = CausalHRM(config).to(device)
    n_params = model.param_count()
    print(f'CausalHRM: {n_params:,} params ({n_params/1e6:.1f}M)')
    print(f'Config: hidden={config.hidden_dim} heads={config.num_heads} '
          f'H_layers={config.num_h_layers} L_layers={config.num_l_layers} '
          f'act_steps={config.max_act_steps} seq_len={config.max_seq_len}')

    if args.compile:
        model = torch.compile(model)

    # Optimizer
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': args.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95), fused=True)

    # Train
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    metrics_log = []
    enc = tiktoken.get_encoding("gpt2")
    t0 = time.time()

    print(f'\n--- Training for {args.max_steps} steps ---')

    step = 0
    train_iter = iter(train_loader)
    while step < args.max_steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        # LR schedule
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # Forward
        with torch.autocast('cuda', dtype=amp_dtype):
            out = model(x, labels=y)
            loss = out['loss']

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        step += 1

        # Log
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            tps = step * args.batch_size * args.max_seq_len / elapsed
            vram = torch.cuda.max_memory_allocated() / (1024**2)
            print(f'  step={step}/{args.max_steps} loss={loss.item():.4f} '
                  f'act_steps={out["act_steps"]} lr={lr:.2e} '
                  f'vram={vram:.0f}MB {tps/1e3:.1f}k tok/s', flush=True)
            metrics_log.append({
                'step': step, 'loss': loss.item(), 'lr': lr,
                'act_steps': out['act_steps'], 'wall_s': elapsed,
            })

        # Eval
        if step % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_loader, device=device, dtype=amp_dtype)
            print(f'  [eval] step={step} val_loss={val_loss:.4f}', flush=True)
            metrics_log.append({
                'step': step, 'val_loss': val_loss, 'wall_s': time.time() - t0,
            })

            # Sample generation
            prompt = "Once upon a time"
            prompt_ids = enc.encode(prompt)
            inp = torch.tensor([prompt_ids], device=device)
            with torch.autocast('cuda', dtype=amp_dtype):
                gen = model.generate(inp, max_new_tokens=100, temperature=0.8, top_k=50)
            gen_text = enc.decode(gen[0].tolist())
            print(f'  [sample] {gen_text[:200]}', flush=True)

        # Save
        if step % args.save_interval == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save({
                'model': raw_model.state_dict(),
                'config': raw_model.config.__dict__,
                'step': step,
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)
            print(f'  Saved checkpoint: {ckpt_path}', flush=True)

    # Final save
    elapsed = time.time() - t0
    ckpt_path = os.path.join(args.checkpoint_dir, f'step_{step}.pt')
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save({
        'model': raw_model.state_dict(),
        'config': raw_model.config.__dict__,
        'step': step,
    }, ckpt_path)

    log_path = os.path.join(args.checkpoint_dir, 'metrics.json')
    with open(log_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)

    print(f'\n{"="*60}')
    print(f'Training complete: {step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)')
    print(f'Final loss: {loss.item():.4f}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Metrics: {log_path}')


if __name__ == '__main__':
    main()
