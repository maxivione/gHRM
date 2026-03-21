import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
print(f'compile: {hasattr(torch, "compile")}')
try:
    import triton
    print(f'Triton: {triton.__version__}')
except ImportError as e:
    print(f'Triton: MISSING ({e})')
import sys
print(f'Python: {sys.version}')
