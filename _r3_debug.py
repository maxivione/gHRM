"""Quick test: does U-Net skip patching + AdamW produce NaN?"""
import sys, os
os.environ['WANDB_MODE'] = 'offline'
os.environ['CC'] = 'gcc'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/lib64:/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['DISABLE_COMPILE'] = '1'
sys.path.insert(0, '/mnt/e/Github/HRM-official')

import torch
import yaml
from pretrain import PretrainConfig, init_train_state, create_dataloader, train_batch

torch.set_float32_matmul_precision('high')

with open('/mnt/e/Github/HRM-official/config/arch/hrm_v1.yaml', 'r') as f:
    arch_cfg = yaml.safe_load(f)
if isinstance(arch_cfg.get('puzzle_emb_ndim'), str):
    arch_cfg['puzzle_emb_ndim'] = arch_cfg['hidden_size']

config = PretrainConfig(
    arch=arch_cfg,
    data_path='/mnt/e/Github/HRM-official/data/sudoku-extreme-1k-aug-1000',
    global_batch_size=384, epochs=10, eval_interval=10,
    lr=7e-5, lr_min_ratio=1.0, lr_warmup_steps=2000,
    puzzle_emb_lr=7e-5, puzzle_emb_weight_decay=1.0,
    weight_decay=1.0, beta1=0.9, beta2=0.95,
)

train_loader, train_metadata = create_dataloader(
    config, 'train', test_set_mode=False,
    epochs_per_iter=10, global_batch_size=384, rank=0, world_size=1
)

train_state = init_train_state(config, train_metadata, world_size=1)

# Test 10 steps with vanilla AdamW, no patches
import math
print("--- Test: vanilla (no patches) ---")
for i, (set_name, batch, gbs) in enumerate(train_loader):
    if i >= 10:
        break
    m = train_batch(config, train_state, batch, gbs, rank=0, world_size=1)
    if m:
        loss = m.get('train/lm_loss', float('nan'))
        print(f"  step {i+1}: lm_loss={loss:.4f} nan={math.isnan(loss)}")

print("DONE")
