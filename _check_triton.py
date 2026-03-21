import torch
print(f"torch.compile available: {hasattr(torch, 'compile')}")
try:
    import triton
    print("triton: YES")
except ImportError as e:
    print(f"triton: NO ({e})")
print(f"Python: {__import__('sys').version}")
