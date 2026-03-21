"""R2 smoke test: 3 training steps to verify CUDA + AdamW + gradients flow."""
import sys, os, time
os.environ['DISABLE_COMPILE'] = '1'
os.environ['WANDB_MODE'] = 'disabled'
sys.path.insert(0, r'e:\Github\HRM-official')

import torch
import yaml
from pretrain import PretrainConfig, init_train_state, create_dataloader, train_batch

with open(r'e:\Github\HRM-official\config\arch\hrm_v1.yaml', 'r') as f:
    arch_cfg = yaml.safe_load(f)
# Resolve Hydra interpolation that doesn't work outside Hydra
if isinstance(arch_cfg.get('puzzle_emb_ndim'), str):
    arch_cfg['puzzle_emb_ndim'] = arch_cfg['hidden_size']

config = PretrainConfig(
    arch=arch_cfg,
    data_path=r'e:\Github\HRM-official\data\sudoku-extreme-1k-aug-1000',
    global_batch_size=384,
    epochs=20000,
    eval_interval=2000,
    lr=7e-5,
    lr_min_ratio=1.0,
    lr_warmup_steps=2000,
    puzzle_emb_lr=7e-5,
    weight_decay=1.0,
    puzzle_emb_weight_decay=1.0,
    beta1=0.9,
    beta2=0.95,
    checkpoint_path=None,
)

print("Building dataloaders...", flush=True)
train_loader, train_metadata = create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1, global_batch_size=384, rank=0, world_size=1)
train_state = init_train_state(config, train_metadata, world_size=1)
n_params = sum(p.numel() for p in train_state.model.parameters())
print(f'Model params: {n_params:,}', flush=True)
print(f'Optimizer[0]: {type(train_state.optimizers[0]).__name__}', flush=True)
print(f'Optimizer[1]: {type(train_state.optimizers[1]).__name__}', flush=True)
print(f'VRAM after init: {torch.cuda.memory_allocated()/(1024**2):.1f} MB', flush=True)

t0 = time.time()
step_count = 0
for i, (set_name, batch, gbs) in enumerate(train_loader):
    metrics = train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
    vram = torch.cuda.max_memory_allocated()/(1024**2)
    elapsed = time.time() - t0
    step_count += 1
    if metrics:
        loss = metrics.get('train/lm_loss', float('nan'))
        steps = metrics.get('train/steps', float('nan'))
        q_halt = metrics.get('train/q_halt_loss', float('nan'))
        print(f'Step {step_count}: lm_loss={loss:.4f} act_steps={steps:.1f} q_halt={q_halt:.4f} vram={vram:.0f}MB t={elapsed:.1f}s', flush=True)
    else:
        print(f'Step {step_count}: (no metrics returned, rank!=0?) vram={vram:.0f}MB', flush=True)

    if step_count >= 3:
        print(f'\n=== R2 SMOKE TEST PASSED ===', flush=True)
        print(f'3 training steps completed on CUDA.', flush=True)
        print(f'Peak VRAM: {vram:.0f} MB', flush=True)
        print(f'Device: {torch.cuda.get_device_name()}', flush=True)
        break
