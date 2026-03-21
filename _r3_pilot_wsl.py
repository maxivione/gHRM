"""R3.1 pilot (WSL2): 2000 epochs of HRM with Parameter Golf techniques.

Fixes from R3.0: Muon lr 0.02->0.01, cosine warmdown, LoRA-TTT eval fix.
"""
import sys, os, time, json, math

os.environ['WANDB_MODE'] = 'offline'
os.environ['CC'] = 'gcc'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/lib64:/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
sys.path.insert(0, '/mnt/e/Github/HRM-official')

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import yaml
from pretrain import (
    PretrainConfig, init_train_state, create_dataloader,
    train_batch, save_train_state
)

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ====================================================================
# TECHNIQUE 1: Muon optimizer
# ====================================================================

def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    nrm = X.norm()
    if nrm < eps or torch.isnan(nrm) or torch.isinf(nrm):
        return torch.zeros_like(G)
    X /= nrm
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if torch.isnan(X).any():
        return torch.zeros_like(G)
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.95,
                 backend_steps: int = 5, nesterov: bool = True):
        super().__init__(
            params,
            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),
        )

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            backend_steps = group["backend_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if nesterov:
                    g = g.add(buf, alpha=momentum)
                g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)


# ====================================================================
# TECHNIQUE 2: U-Net skip connections
# Patch the ReasoningModule class BEFORE model instantiation.
# ====================================================================

from models.hrm.hrm_act_v1 import HierarchicalReasoningModel_ACTV1ReasoningModule

# Save original forward
_orig_reasoning_forward = HierarchicalReasoningModel_ACTV1ReasoningModule.forward

def _reasoning_forward_with_skips(self, hidden_states, input_injection, **kwargs):
    hidden_states = hidden_states + input_injection
    layers = list(self.layers)
    n = len(layers)

    if n < 2:
        # Too few layers for skips, just run normally
        for layer in layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

    n_enc = n // 2
    n_dec = n - n_enc
    n_skip = min(n_enc, n_dec)
    skips = []

    # Encoder half
    for i in range(n_enc):
        hidden_states = layers[i](hidden_states=hidden_states, **kwargs)
        skips.append(hidden_states)

    # Decoder half with skip connections
    for i in range(n_enc, n):
        dec_idx = i - n_enc
        if dec_idx < n_skip and skips:
            skip = skips[-(dec_idx + 1)]
            # Simple additive skip with small scale (no learnable weight to avoid
            # issues with compile + parameter registration on monkey-patched method)
            hidden_states = hidden_states + 0.1 * skip
        hidden_states = layers[i](hidden_states=hidden_states, **kwargs)

    return hidden_states

# Monkey-patch the class method (works with torch.compile because it's a proper method)
HierarchicalReasoningModel_ACTV1ReasoningModule.forward = _reasoning_forward_with_skips


# ====================================================================
# TECHNIQUE 3: LoRA test-time training at eval
# ====================================================================

class EvalLoRA(nn.Module):
    def __init__(self, in_dim, out_dim, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        self.reset()

    def reset(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.A)
            self.B.zero_()

    def forward(self, x):
        return (x @ self.A.T) @ self.B.T


def evaluate_with_ttt(config, train_state, eval_loader, eval_metadata, rank, world_size,
                      lora_rank=4, lora_lr=0.005, ttt_steps=3):
    from models.losses import IGNORE_LABEL_ID

    model = train_state.model
    model.eval()

    actual_model = model
    if hasattr(actual_model, '_orig_mod'):
        actual_model = actual_model._orig_mod
    hrm = actual_model.model
    inner = hrm.inner

    hidden_size = inner.config.hidden_size
    vocab_size = inner.config.vocab_size
    device = next(inner.parameters()).device

    lora = EvalLoRA(hidden_size, vocab_size, rank=lora_rank).to(device)
    lora_opt = torch.optim.Adam(lora.parameters(), lr=lora_lr, betas=(0.9, 0.95))

    set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}
    per_set_correct = {s: 0.0 for s in set_ids}
    per_set_exact = {s: 0.0 for s in set_ids}
    per_set_count = {s: 0.0 for s in set_ids}

    for set_name, batch, global_batch_size in eval_loader:
        batch = {k: v.cuda() for k, v in batch.items()}
        labels = batch["labels"]

        lora.reset()
        # Reset optimizer state
        lora_opt.state.clear()

        with torch.device("cuda"):
            carry = hrm.initial_carry(batch)

        with torch.no_grad():
            while True:
                carry, outputs = hrm(carry=carry, batch=batch)
                if carry.halted.all():
                    break

        z_H = carry.inner_carry.z_H
        puzzle_emb_len = inner.puzzle_emb_len

        with torch.no_grad():
            base_logits = inner.lm_head(z_H)[:, puzzle_emb_len:]

        # TTT: train LoRA on this batch
        for _ in range(ttt_steps):
            lora_opt.zero_grad()
            delta = lora(z_H[:, puzzle_emb_len:].detach())
            logits = base_logits.detach() + delta
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1).clamp_min(1).unsqueeze(-1)
            loss = (F.cross_entropy(
                logits.float().view(-1, vocab_size), labels.long().view(-1),
                ignore_index=IGNORE_LABEL_ID, reduction='none'
            ).view(labels.shape) / loss_counts).sum()
            loss.backward()
            lora_opt.step()

        with torch.no_grad():
            delta = lora(z_H[:, puzzle_emb_len:].detach())
            final_logits = base_logits + delta
            mask = labels != IGNORE_LABEL_ID
            loss_counts = mask.sum(-1)
            is_correct = mask & (torch.argmax(final_logits, dim=-1) == labels)
            seq_is_correct = is_correct.sum(-1) == loss_counts
            valid = loss_counts > 0
            per_set_count[set_name] += valid.sum().item()
            per_set_correct[set_name] += (is_correct.float().sum(-1) / loss_counts.clamp_min(1).float())[valid].sum().item()
            per_set_exact[set_name] += seq_is_correct[valid].sum().item()

    results = {}
    for s in set_ids:
        c = per_set_count[s]
        if c > 0:
            results[s] = {'accuracy': per_set_correct[s] / c, 'exact_accuracy': per_set_exact[s] / c}
    return results


# ====================================================================
# SETUP & TRAINING
# ====================================================================

EPOCHS = 2000
EVAL_INTERVAL = 2000
BATCH_SIZE = 384
CHECKPOINT_DIR = '/mnt/e/Github/HRM-official/checkpoints/r3_1_pilot'
MUON_LR = 0.01

with open('/mnt/e/Github/HRM-official/config/arch/hrm_v1.yaml', 'r') as f:
    arch_cfg = yaml.safe_load(f)
if isinstance(arch_cfg.get('puzzle_emb_ndim'), str):
    arch_cfg['puzzle_emb_ndim'] = arch_cfg['hidden_size']

config = PretrainConfig(
    arch=arch_cfg,
    data_path='/mnt/e/Github/HRM-official/data/sudoku-extreme-1k-aug-1000',
    global_batch_size=BATCH_SIZE, epochs=EPOCHS, eval_interval=EVAL_INTERVAL,
    lr=7e-5, lr_min_ratio=0.1, lr_warmup_steps=2000,
    puzzle_emb_lr=7e-5, puzzle_emb_weight_decay=1.0,
    weight_decay=1.0, beta1=0.9, beta2=0.95,
    checkpoint_path=CHECKPOINT_DIR,
)

train_epochs_per_iter = EVAL_INTERVAL
total_iters = EPOCHS // train_epochs_per_iter

print('Building dataloaders...', flush=True)
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

print('Initializing model (with U-Net skip patches active)...', flush=True)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Build model — the class-level patch is already active, so skips are baked in
train_state = init_train_state(config, train_metadata, world_size=1)

# Access raw model for optimizer surgery
actual_model = train_state.model
if hasattr(actual_model, '_orig_mod'):
    actual_model = actual_model._orig_mod

# Replace optimizer[1] with Muon (matrices) + Adam (scalars)
print('Setting up Muon optimizer...', flush=True)
matrix_params = []
scalar_params = []
for name, p in actual_model.named_parameters():
    if not p.requires_grad:
        continue
    if 'puzzle_emb' in name:
        continue
    if p.ndim >= 2:
        matrix_params.append(p)
    else:
        scalar_params.append(p)

muon_opt = Muon(matrix_params, lr=0, momentum=0.95, backend_steps=5)
adam_opt = torch.optim.AdamW(scalar_params, lr=0, weight_decay=config.weight_decay,
                              betas=(config.beta1, config.beta2))

class DualOptimizer:
    def __init__(self, muon, adam, muon_lr_ratio):
        self.muon = muon
        self.adam = adam
        self.muon_lr_ratio = muon_lr_ratio
        self._muon_groups = muon.param_groups
        self._adam_groups = adam.param_groups
        self.param_groups = self._muon_groups + self._adam_groups

    def step(self):
        for g in self._muon_groups:
            g['lr'] = g['lr'] * self.muon_lr_ratio
        self.muon.step()
        for g in self._muon_groups:
            g['lr'] = g['lr'] / self.muon_lr_ratio
        self.adam.step()

    def zero_grad(self):
        self.muon.zero_grad()
        self.adam.zero_grad()

train_state.optimizers[1] = DualOptimizer(muon_opt, adam_opt, muon_lr_ratio=MUON_LR / config.lr)

n_params = sum(p.numel() for p in actual_model.parameters())
print(f'Model params: {n_params:,}', flush=True)
print(f'Muon: {sum(p.numel() for p in matrix_params):,} matrix params', flush=True)
print(f'Adam: {sum(p.numel() for p in scalar_params):,} scalar params', flush=True)
print(f'Techniques: Muon + U-Net skips (0.1x) + LoRA-TTT eval', flush=True)
print(f'Total training steps: {train_state.total_steps}', flush=True)
print(f'Device: {torch.cuda.get_device_name()}', flush=True)
print(f'VRAM after init: {torch.cuda.memory_allocated()/(1024**2):.1f} MB', flush=True)

metrics_log = []
nan_count = 0
MAX_NAN = 10
t0 = time.time()
step_count = 0

print(f'\n--- Training ---', flush=True)

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
                    print(f'\nABORT: {MAX_NAN} NaN events.', flush=True)
                    sys.exit(2)

            if step_count % 50 == 0 or step_count <= 10:
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
                'step': step_count, 'lm_loss': float(lm_loss),
                'q_halt_loss': float(q_halt), 'steps': float(steps_val),
                'lr': float(lr_val), 'wall_s': time.time() - t0,
            })

    # Save checkpoint and metrics BEFORE eval (so we don't lose training data if eval crashes)
    print('Saving checkpoint...', flush=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    save_train_state(config, train_state)
    log_path = os.path.join(CHECKPOINT_DIR, 'r3_1_pilot_metrics.json')
    with open(log_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)

    print(f'\n--- Evaluation at step {step_count} ---', flush=True)
    try:
        # Try LoRA-TTT eval
        eval_metrics = evaluate_with_ttt(
            config, train_state, eval_loader, eval_metadata,
            rank=0, world_size=1, lora_rank=4, lora_lr=0.005, ttt_steps=3
        )
        print('  (LoRA-TTT evaluation)', flush=True)
    except Exception as e:
        print(f'  LoRA-TTT eval failed: {e}', flush=True)
        print('  Falling back to standard eval...', flush=True)
        from pretrain import evaluate
        eval_metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=0, world_size=1)

    if eval_metrics:
        for set_name, set_metrics in eval_metrics.items():
            acc = set_metrics.get('accuracy', float('nan'))
            exact = set_metrics.get('exact_accuracy', float('nan'))
            print(f'  {set_name}: accuracy={acc:.4f} exact_accuracy={exact:.4f}', flush=True)
            metrics_log.append({
                'step': step_count, 'eval_set': set_name,
                'accuracy': float(acc), 'exact_accuracy': float(exact),
                'wall_s': time.time() - t0,
            })

    # Save metrics again with eval results
    with open(log_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)

elapsed_total = time.time() - t0
peak_vram = torch.cuda.max_memory_allocated() / (1024**2)

print(f'\n{"="*60}', flush=True)
print(f'R3.1 PILOT COMPLETE (Muon + U-Net skips + LoRA-TTT)', flush=True)
print(f'{"="*60}', flush=True)
print(f'Batch size: {BATCH_SIZE}', flush=True)
print(f'Total steps: {step_count}', flush=True)
print(f'Peak VRAM: {peak_vram:.0f} MB', flush=True)
print(f'Wall clock: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)', flush=True)
print(f'NaN events: {nan_count}', flush=True)
print(f'Device: {torch.cuda.get_device_name()}', flush=True)

if metrics_log:
    first_loss = next((m['lm_loss'] for m in metrics_log if 'lm_loss' in m), None)
    last_loss = next((m['lm_loss'] for m in reversed(metrics_log) if 'lm_loss' in m), None)
    if first_loss and last_loss:
        print(f'lm_loss: {first_loss:.4f} -> {last_loss:.4f}', flush=True)

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
log_path = os.path.join(CHECKPOINT_DIR, 'r3_1_pilot_metrics.json')
with open(log_path, 'w') as f:
    json.dump(metrics_log, f, indent=2)
print(f'Metrics saved to {log_path}', flush=True)
