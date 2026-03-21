"""Interactive sudoku solver using the trained R3.1 HRM model.

Enter a sudoku puzzle as 81 digits (0 = empty cell), and watch the model
reason through the ACT loop to solve it.
"""
import sys, os

os.environ['WANDB_MODE'] = 'offline'
os.environ['CC'] = 'gcc'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/lib64:/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['DISABLE_COMPILE'] = '1'  # skip compile for interactive use

sys.path.insert(0, '/mnt/e/Github/HRM-official')

import torch
import yaml
import numpy as np
from pretrain import PretrainConfig, init_train_state, create_dataloader

CHECKPOINT = '/mnt/e/Github/HRM-official/checkpoints/r3_1_pilot/step_5208'
torch.set_float32_matmul_precision('high')


def load_model():
    with open('/mnt/e/Github/HRM-official/config/arch/hrm_v1.yaml', 'r') as f:
        arch_cfg = yaml.safe_load(f)
    if isinstance(arch_cfg.get('puzzle_emb_ndim'), str):
        arch_cfg['puzzle_emb_ndim'] = arch_cfg['hidden_size']

    config = PretrainConfig(
        arch=arch_cfg,
        data_path='/mnt/e/Github/HRM-official/data/sudoku-extreme-1k-aug-1000',
        global_batch_size=1, epochs=1, eval_interval=1,
        lr=1e-5, lr_min_ratio=1.0, lr_warmup_steps=1,
        puzzle_emb_lr=1e-5, puzzle_emb_weight_decay=1.0,
        weight_decay=1.0, beta1=0.9, beta2=0.95,
        checkpoint_path=os.path.dirname(CHECKPOINT),
    )

    train_loader, train_metadata = create_dataloader(
        config, 'train', test_set_mode=False,
        epochs_per_iter=1, global_batch_size=1, rank=0, world_size=1
    )

    train_state = init_train_state(config, train_metadata, world_size=1)

    # Load checkpoint
    state_dict = torch.load(CHECKPOINT, map_location='cuda')
    # Strip _orig_mod. prefix if present (from torch.compile)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.removeprefix('_orig_mod.')] = v

    try:
        train_state.model.load_state_dict(cleaned, assign=True)
    except RuntimeError:
        train_state.model.load_state_dict(state_dict, assign=True)

    train_state.model.eval()
    train_state.model.cuda()
    return train_state.model


def parse_puzzle(text):
    """Parse a puzzle string like '003020600900305001001806400...' into a batch."""
    digits = ''.join(c for c in text if c.isdigit() or c == '.')
    digits = digits.replace('.', '0')
    if len(digits) != 81:
        return None
    # Vocab: 0=PAD, 1-9=digits 0-8, 10=digit 9
    # Actually from build_sudoku: values are shifted +1, so digit 0 -> token 1, digit 9 -> token 10
    arr = np.array([int(c) for c in digits], dtype=np.int64).reshape(1, 81) + 1
    return arr


def print_grid(tokens, highlight_solved=None):
    """Pretty-print a 9x9 sudoku grid from token array."""
    grid = (tokens - 1).reshape(9, 9)  # Convert back from tokens to digits
    print('┌───────┬───────┬───────┐')
    for r in range(9):
        if r > 0 and r % 3 == 0:
            print('├───────┼───────┼───────┤')
        row = '│'
        for c in range(9):
            if c > 0 and c % 3 == 0:
                row += '│'
            val = grid[r, c]
            idx = r * 9 + c
            if val == 0:
                row += ' . '
            elif highlight_solved is not None and highlight_solved[idx]:
                row += f' \033[92m{val}\033[0m '  # Green for solved cells
            else:
                row += f' {val} '
        row += '│'
        print(row)
    print('└───────┴───────┴───────┘')


def solve(model, puzzle_tokens):
    """Run the HRM ACT loop to solve a sudoku puzzle."""
    device = next(model.parameters()).device
    batch = {
        'inputs': torch.tensor(puzzle_tokens, dtype=torch.long, device=device),
        'labels': torch.zeros(1, 81, dtype=torch.long, device=device),
        'puzzle_identifiers': torch.zeros(1, dtype=torch.long, device=device),
    }

    # Get the HRM model (might be wrapped)
    hrm = model
    if hasattr(hrm, '_orig_mod'):
        hrm = hrm._orig_mod
    if hasattr(hrm, 'model'):
        hrm = hrm.model

    with torch.device(device):
        carry = hrm.initial_carry(batch)

    print('\n🧠 Reasoning...')
    step = 0
    with torch.no_grad():
        while True:
            carry, outputs = hrm(carry=carry, batch=batch)
            step += 1

            logits = outputs['logits']  # (1, 81, vocab_size)
            preds = logits.argmax(dim=-1)[0]  # (81,)
            confidence = torch.softmax(logits[0], dim=-1).max(dim=-1).values.mean().item()

            print(f'  Step {step}: avg confidence={confidence:.3f}', end='')
            if carry.halted.all():
                print(' [HALTED]')
                break
            print()

            if step >= 32:
                print('  (max steps reached)')
                break

    # Get final predictions
    final_logits = outputs['logits']
    predicted = final_logits.argmax(dim=-1)[0].cpu().numpy()  # (81,) in token space

    # Merge: keep given cells, fill in predicted for blanks
    original = puzzle_tokens[0]
    was_blank = (original == 1)  # Token 1 = digit 0 = blank
    solution = original.copy()
    solution[was_blank] = predicted[was_blank]

    return solution, was_blank, step


def main():
    print('Loading model...')
    model = load_model()
    print('Model loaded! ✓\n')

    # Example puzzles
    examples = {
        'easy':   '003020600900305001001806400008102900700000008006708200002609500800203009005010300',
        'medium': '200080300060070084030500209000105408000000000402706000301007040720040060004010003',
        'hard':   '800000000003600000070090200050007000000045700000100030001000068008500010090000400',
    }

    print('=' * 50)
    print('  HRM Sudoku Solver (R3.1 — Muon + U-Net Skips)')
    print('=' * 50)
    print()
    print('Enter a sudoku puzzle as 81 digits (0 or . for blanks)')
    print('Or type "easy", "medium", "hard" for examples')
    print('Type "quit" to exit\n')

    while True:
        try:
            text = input('Puzzle> ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not text or text == 'quit':
            break

        if text in examples:
            text = examples[text]
            print(f'Using {text[:20]}...')

        tokens = parse_puzzle(text)
        if tokens is None:
            print('Invalid puzzle. Need exactly 81 digits (0-9 or "." for blanks).')
            continue

        print('\n📋 Input puzzle:')
        print_grid(tokens[0])

        solution, was_blank, steps = solve(model, tokens)

        print(f'\n✅ Solution ({steps} ACT steps):')
        print_grid(solution, highlight_solved=was_blank)

        # Verify
        grid = (solution - 1).reshape(9, 9)
        valid = True
        for i in range(9):
            row = set(grid[i, :])
            col = set(grid[:, i])
            box = set(grid[(i//3)*3:(i//3)*3+3, (i%3)*3:(i%3)*3+3].flatten())
            if row != set(range(1, 10)) or col != set(range(1, 10)) or box != set(range(1, 10)):
                valid = False
                break

        if valid:
            print('✓ Valid sudoku solution!')
        else:
            print('✗ Solution has errors (model may need more training)')
        print()


if __name__ == '__main__':
    main()
