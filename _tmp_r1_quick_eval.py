"""Quick checkpoint eval: runs only a few batches to verify metrics are correct."""
import os, sys, yaml, time, json
os.environ['DISABLE_COMPILE'] = '1'
os.environ['WANDB_MODE'] = 'disabled'

sys.path.insert(0, r'e:\Github\HRM-official')

import torch
from pretrain import PretrainConfig, init_train_state, create_dataloader
from models.losses import IGNORE_LABEL_ID

with open(r'e:\Github\HRM-official\checkpoints\sudoku-extreme\all_config.yaml', 'r') as f:
    config = PretrainConfig(**yaml.safe_load(f))
config.checkpoint_path = r'e:\Github\HRM-official\checkpoints\sudoku-extreme'
config.eval_save_outputs = []

train_loader, train_metadata = create_dataloader(config, 'train', test_set_mode=False, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)
eval_loader, eval_metadata = create_dataloader(config, 'test', test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=0, world_size=1)

train_state = init_train_state(config, train_metadata, world_size=1)
try:
    train_state.model.load_state_dict(torch.load(r'e:\Github\HRM-official\checkpoints\sudoku-extreme\checkpoint', map_location='cuda'), assign=True)
except:
    train_state.model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in torch.load(r'e:\Github\HRM-official\checkpoints\sudoku-extreme\checkpoint', map_location='cuda').items()}, assign=True)

train_state.step = 0
train_state.model.eval()
print(f'Model loaded. VRAM: {torch.cuda.memory_allocated()/(1024**2):.1f} MB', flush=True)

MAX_BATCHES = 5
total_correct = 0
total_tokens = 0
batch_count = 0
t0 = time.time()

with torch.inference_mode():
    for set_name, batch, global_batch_size in eval_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.device("cuda"):
            carry = train_state.model.initial_carry(batch)

        step = 0
        while True:
            carry, _, metrics, preds, all_finish = train_state.model(carry=carry, batch=batch, return_keys=['logits'])
            step += 1
            if all_finish:
                break

        logits = preds['logits']
        labels = batch['labels']
        pred_ids = logits.argmax(dim=-1)
        mask = labels != IGNORE_LABEL_ID
        correct = (pred_ids[mask] == labels[mask]).sum().item()
        total = mask.sum().item()
        total_correct += correct
        total_tokens += total

        batch_count += 1
        elapsed = time.time() - t0
        acc = total_correct / max(total_tokens, 1)
        print(f'Batch {batch_count}/{MAX_BATCHES}: acc={acc:.4f} ({total_correct}/{total_tokens}) steps={step} elapsed={elapsed:.1f}s', flush=True)
        for mk, mv in sorted(metrics.items()):
            print(f'  {mk}: {float(mv):.6f}', flush=True)

        if batch_count >= MAX_BATCHES:
            break

elapsed = time.time() - t0
accuracy = total_correct / max(total_tokens, 1)
peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

result = {
    'batches_evaluated': batch_count,
    'accuracy': round(accuracy, 6),
    'total_correct': total_correct,
    'total_tokens': total_tokens,
    'elapsed_seconds': round(elapsed, 1),
    'peak_vram_mb': round(peak_vram, 1),
}
print(flush=True)
print('=== R1 Quick Eval Results ===', flush=True)
for k, v in result.items():
    print(f'{k}: {v}', flush=True)

with open(r'e:\Github\gHRM\r1_quick_eval_results.json', 'w') as f:
    json.dump(result, f, indent=2)
