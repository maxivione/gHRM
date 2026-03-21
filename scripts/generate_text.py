"""Interactive text generation with trained CausalHRM.

Usage:
    python scripts/generate_text.py --checkpoint checkpoints/causal_hrm_tinystories/step_10000.pt
"""
import os
import sys
import argparse

import torch
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.causal_hrm import CausalHRM, CausalHRMConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--temperature', type=float, default=0.8)
    p.add_argument('--top_k', type=int, default=50)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--max_new_tokens', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    enc = tiktoken.get_encoding("gpt2")

    # Load model
    print('Loading model...')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = CausalHRMConfig(**ckpt['config'])
    model = CausalHRM(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    n_params = model.param_count()
    print(f'CausalHRM loaded: {n_params/1e6:.1f}M params')
    print(f'Act steps: {config.max_act_steps}, Seq len: {config.max_seq_len}')
    print()
    print('='*50)
    print('  gHRM Text Generation')
    print('='*50)
    print()
    print('Type a prompt and press Enter to generate text.')
    print('Type "quit" to exit.')
    print()

    while True:
        try:
            prompt = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not prompt or prompt == 'quit':
            break

        tokens = enc.encode(prompt)
        input_ids = torch.tensor([tokens], device=device)

        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )

        generated = enc.decode(output_ids[0].tolist())
        print(f'\n{generated}\n')


if __name__ == '__main__':
    main()
