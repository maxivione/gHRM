import os
os.environ['CC'] = 'gcc'
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.8'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/lib64:/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TORCH_LOGS'] = '+dynamo'
os.environ['TORCHDYNAMO_VERBOSE'] = '1'

import torch

print("Testing torch.compile inductor on CUDA...")
try:
    @torch.compile(backend="inductor")
    def test_fn(x):
        return x @ x + x

    x = torch.randn(8, 8, device="cuda")
    result = test_fn(x)
    print(f"SUCCESS: result shape={result.shape}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}")
    import traceback
    traceback.print_exc()
