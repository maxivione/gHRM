"""R2 speed benchmark: 4 conditions x 20 steps each.
Conditions:
  A) baseline (no compile, no TF32)    — current setup
  B) TF32 only
  C) compile only
  D) compile + TF32
"""
import sys, os, time, gc
sys.path.insert(0, r'e:\Github\HRM-official')
os.environ['WANDB_MODE'] = 'disabled'

import torch
import yaml
from pretrain import PretrainConfig, init_train_state, create_dataloader, train_batch

STEPS_PER_CONDITION = 20

with open(r'e:\Github\HRM-official\config\arch\hrm_v1.yaml', 'r') as f:
    arch_cfg = yaml.safe_load(f)
if isinstance(arch_cfg.get('puzzle_emb_ndim'), str):
    arch_cfg['puzzle_emb_ndim'] = arch_cfg['hidden_size']


def run_condition(label, use_compile, use_tf32):
    print(f'\n{"="*60}', flush=True)
    print(f'CONDITION: {label}', flush=True)
    print(f'  compile={use_compile}, tf32={use_tf32}', flush=True)
    print(f'{"="*60}', flush=True)

    # TF32 toggle
    if use_tf32:
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    # Compile toggle
    if use_compile:
        os.environ.pop('DISABLE_COMPILE', None)
    else:
        os.environ['DISABLE_COMPILE'] = '1'

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()

    config = PretrainConfig(
        arch=arch_cfg,
        data_path=r'e:\Github\HRM-official\data\sudoku-extreme-1k-aug-1000',
        global_batch_size=384,
        epochs=2000,
        eval_interval=2000,
        lr=7e-5,
        lr_min_ratio=1.0,
        lr_warmup_steps=2000,
        puzzle_emb_lr=7e-5,
        puzzle_emb_weight_decay=1.0,
        weight_decay=1.0,
        beta1=0.9,
        beta2=0.95,
        checkpoint_path=None,
    )

    train_loader, train_metadata = create_dataloader(
        config, 'train', test_set_mode=False,
        epochs_per_iter=2000,
        global_batch_size=384, rank=0, world_size=1
    )
    train_state = init_train_state(config, train_metadata, world_size=1)

    print(f'  Init VRAM: {torch.cuda.memory_allocated()/(1024**2):.0f} MB', flush=True)

    step_times = []
    losses = []
    for i, (set_name, batch, gbs) in enumerate(train_loader):
        t0 = time.perf_counter()
        metrics = train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

        step_times.append(dt)
        if metrics:
            losses.append(metrics.get('train/lm_loss', float('nan')))

        if (i + 1) >= STEPS_PER_CONDITION:
            break

    peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

    # Skip first 2 steps (compile warmup / cache warmup)
    warmup = 2
    timed = step_times[warmup:]
    avg = sum(timed) / len(timed) if timed else 0
    projected_hours = avg * 5208 / 3600

    print(f'\n  Results for {label}:', flush=True)
    print(f'    First 3 step times: {[f"{t:.2f}s" for t in step_times[:3]]}', flush=True)
    print(f'    Avg step (after warmup): {avg:.2f}s', flush=True)
    print(f'    Projected total: {projected_hours:.1f}h for 5208 steps', flush=True)
    print(f'    Peak VRAM: {peak_vram:.0f} MB', flush=True)
    print(f'    Loss trajectory: {losses[0]:.4f} -> {losses[-1]:.4f}', flush=True)

    del train_state, train_loader
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'label': label,
        'avg_step_s': avg,
        'projected_h': projected_hours,
        'peak_vram_mb': peak_vram,
        'step_times': step_times,
        'first_loss': losses[0] if losses else None,
        'last_loss': losses[-1] if losses else None,
    }


results = []

# A: baseline
results.append(run_condition('A: baseline', use_compile=False, use_tf32=False))

# B: TF32 only
results.append(run_condition('B: TF32 only', use_compile=False, use_tf32=True))

# C: compile only
try:
    results.append(run_condition('C: compile only', use_compile=True, use_tf32=False))
except Exception as e:
    print(f'\n  COMPILE FAILED: {e}', flush=True)
    results.append({'label': 'C: compile only', 'avg_step_s': None, 'error': str(e)})

# D: compile + TF32
try:
    results.append(run_condition('D: compile + TF32', use_compile=True, use_tf32=True))
except Exception as e:
    print(f'\n  COMPILE+TF32 FAILED: {e}', flush=True)
    results.append({'label': 'D: compile + TF32', 'avg_step_s': None, 'error': str(e)})

print(f'\n\n{"="*60}', flush=True)
print(f'SUMMARY', flush=True)
print(f'{"="*60}', flush=True)
for r in results:
    if r.get('avg_step_s') is not None:
        print(f"  {r['label']:25s}  {r['avg_step_s']:.2f}s/step  {r['projected_h']:.1f}h  {r['peak_vram_mb']:.0f}MB VRAM", flush=True)
    else:
        print(f"  {r['label']:25s}  FAILED: {r.get('error', '?')}", flush=True)
