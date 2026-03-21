"""R2 pilot: 2000 epochs of official HRM on Sudoku-Extreme 1K.

Deviations from official:
  - AdamW instead of adam_atan2 (CUDA-only on Windows)
  - SDPA instead of flash_attn (patched in layers.py)
  - num_workers=0 (Windows multiprocessing)
  - TF32 enabled (Ampere low-risk precision — 32% speedup measured)
  - WANDB offline
  - torch.compile disabled (no Triton on Windows)
"""
import sys, os, time, json, math

os.environ['DISABLE_COMPILE'] = '1'
os.environ['WANDB_MODE'] = 'offline'
sys.path.insert(0, r'e:\Github\HRM-official')

import torch
import yaml
from pretrain import (
    PretrainConfig, init_train_state, create_dataloader,
    train_batch, evaluate, save_train_state
)

# TF32 — measured 32% faster, same VRAM, negligible precision impact
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

EPOCHS = 2000
EVAL_INTERVAL = 2000
BATCH_SIZE = 384
CHECKPOINT_DIR = r'e:\Github\HRM-official\checkpoints\r2_pilot'

with open(r'e:\Github\HRM-official\config\arch\hrm_v1.yaml', 'r') as f:
    arch_cfg = yaml.safe_load(f)
if isinstance(arch_cfg.get('puzzle_emb_ndim'), str):
    arch_cfg['puzzle_emb_ndim'] = arch_cfg['hidden_size']

config = PretrainConfig(
    arch=arch_cfg,
    data_path=r'e:\Github\HRM-official\data\sudoku-extreme-1k-aug-1000',
    global_batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    eval_interval=EVAL_INTERVAL,
    lr=7e-5,
    lr_min_ratio=1.0,
    lr_warmup_steps=2000,
    puzzle_emb_lr=7e-5,
    puzzle_emb_weight_decay=1.0,
    weight_decay=1.0,
    beta1=0.9,
    beta2=0.95,
    checkpoint_path=CHECKPOINT_DIR,
)

train_epochs_per_iter = EVAL_INTERVAL
total_iters = EPOCHS // train_epochs_per_iter

print(f'Building dataloaders...', flush=True)
train_loader, train_metadata = create_dataloader(
    config, 'train', test_set_mode=False,
    epochs_per_iter=train_epochs_per_iter,
    global_batch_size=BATCH_SIZE, rank=0, world_size=1
)
eval_loader, eval_metadata = create_dataloader(
    config, 'test', test_set_mode=True,
    epochs_per_iter=1,
    global_batch_size=BATCH_SIZE, rank=0, world_size=1
)

print(f'Initializing model...', flush=True)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
train_state = init_train_state(config, train_metadata, world_size=1)

n_params = sum(p.numel() for p in train_state.model.parameters())
print(f'Model params: {n_params:,}', flush=True)
print(f'Optimizer[0]: {type(train_state.optimizers[0]).__name__}', flush=True)
print(f'Optimizer[1]: {type(train_state.optimizers[1]).__name__}', flush=True)
print(f'Total training steps: {train_state.total_steps}', flush=True)
print(f'Device: {torch.cuda.get_device_name()}', flush=True)
print(f'TF32: enabled', flush=True)
print(f'VRAM after init: {torch.cuda.memory_allocated()/(1024**2):.1f} MB', flush=True)

# Tracking
metrics_log = []
nan_count = 0
MAX_NAN = 5
t0 = time.time()
step_count = 0

print(f'\n--- Training (TF32 enabled) ---', flush=True)

for iter_id in range(total_iters):
    epoch_start = iter_id * train_epochs_per_iter
    print(f'\nEpoch block {epoch_start}-{epoch_start + train_epochs_per_iter}', flush=True)

    train_state.model.train()
    for set_name, batch, gbs in train_loader:
        metrics = train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
        step_count += 1

        if metrics:
            lm_loss = metrics.get('train/lm_loss', float('nan'))
            q_halt = metrics.get('train/q_halt_loss', float('nan'))
            steps_val = metrics.get('train/steps', float('nan'))
            lr_val = metrics.get('train/lr', 0)

            if math.isnan(lm_loss) or math.isnan(q_halt):
                nan_count += 1
                print(f'  WARNING: NaN at step {step_count} (count={nan_count}/{MAX_NAN})', flush=True)
                if nan_count >= MAX_NAN:
                    print(f'\nABORT: {MAX_NAN} NaN events. Stopping.', flush=True)
                    sys.exit(2)

            # Log every 50 steps + first 5
            if step_count % 50 == 0 or step_count <= 5:
                vram = torch.cuda.max_memory_allocated() / (1024**2)
                elapsed = time.time() - t0
                steps_per_sec = step_count / elapsed
                eta_h = (train_state.total_steps - step_count) / steps_per_sec / 3600
                print(
                    f'  step={step_count}/{train_state.total_steps} '
                    f'lm_loss={lm_loss:.4f} q_halt={q_halt:.4f} '
                    f'steps={steps_val:.1f} lr={lr_val:.2e} '
                    f'vram={vram:.0f}MB '
                    f'{steps_per_sec:.2f}it/s ETA={eta_h:.1f}h',
                    flush=True
                )

            metrics_log.append({
                'step': step_count,
                'lm_loss': float(lm_loss),
                'q_halt_loss': float(q_halt),
                'steps': float(steps_val),
                'lr': float(lr_val),
                'wall_s': time.time() - t0,
            })

    # Eval
    print(f'\n--- Evaluation at step {step_count} ---', flush=True)
    train_state.model.eval()
    eval_metrics = evaluate(
        config, train_state, eval_loader, eval_metadata, rank=0, world_size=1
    )
    if eval_metrics:
        for set_name, set_metrics in eval_metrics.items():
            acc = set_metrics.get('accuracy', float('nan'))
            exact = set_metrics.get('exact_accuracy', float('nan'))
            print(f'  {set_name}: accuracy={acc:.4f} exact_accuracy={exact:.4f}', flush=True)
            metrics_log.append({
                'step': step_count,
                'eval_set': set_name,
                'accuracy': float(acc),
                'exact_accuracy': float(exact),
                'wall_s': time.time() - t0,
            })

    # Checkpoint
    print(f'Saving checkpoint...', flush=True)
    save_train_state(config, train_state)

# Final summary
elapsed_total = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

print(f'\n{"="*60}', flush=True)
print(f'R2 PILOT COMPLETE', flush=True)
print(f'{"="*60}', flush=True)
print(f'Batch size: {BATCH_SIZE}', flush=True)
print(f'Total steps: {step_count}', flush=True)
print(f'Peak VRAM: {peak_vram:.0f} MB', flush=True)
print(f'Wall clock: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)', flush=True)
print(f'NaN events: {nan_count}', flush=True)
print(f'Device: {torch.cuda.get_device_name()}', flush=True)
print(f'TF32: enabled', flush=True)

if metrics_log:
    first_loss = next((m['lm_loss'] for m in metrics_log if 'lm_loss' in m), None)
    last_loss = next((m['lm_loss'] for m in reversed(metrics_log) if 'lm_loss' in m), None)
    if first_loss and last_loss:
        print(f'lm_loss: {first_loss:.4f} -> {last_loss:.4f}', flush=True)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
log_path = os.path.join(CHECKPOINT_DIR, 'r2_pilot_metrics.json')
with open(log_path, 'w') as f:
    json.dump(metrics_log, f, indent=2)
print(f'Metrics saved to {log_path}', flush=True)
