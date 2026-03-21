import os, subprocess
os.environ['CC'] = 'gcc'
os.environ['PATH'] = '/usr/local/cuda-12.8/bin:' + os.environ.get('PATH', '')

# Manually reproduce what Triton does
import tempfile
src = """
#include <cuda.h>
int main() { return 0; }
"""
with tempfile.NamedTemporaryFile(suffix='.c', mode='w', delete=False) as f:
    f.write(src)
    src_path = f.name

out_path = src_path.replace('.c', '.so')
cmd = ['gcc', src_path, '-shared', '-o', out_path, '-I/usr/local/cuda-12.8/include', '-L/usr/local/cuda-12.8/lib64', '-lcuda']
print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print(f"Return code: {result.returncode}")
if result.stdout: print(f"stdout: {result.stdout}")
if result.stderr: print(f"stderr: {result.stderr}")
