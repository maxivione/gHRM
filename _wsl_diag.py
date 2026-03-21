import torch, time
print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name()}")

# Test compile with inductor
print("Testing torch.compile inductor...")
try:
    @torch.compile(backend="inductor")
    def test_fn(x):
        return x @ x + x
    x = torch.randn(4, 4, device="cuda")
    result = test_fn(x)
    print(f"  inductor: OK")
except Exception as e:
    print(f"  inductor: FAILED ({type(e).__name__}: {e})")

# TF32 matmul speed
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
a = torch.randn(512, 512, device="cuda")
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(1000):
    _ = a @ a
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"1000x [512x512] matmuls (TF32): {dt:.3f}s")
