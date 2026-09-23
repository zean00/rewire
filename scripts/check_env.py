"""Step 0 gate (IMPLEMENTATION_PLAN.md 0.2/0.3): print toolchain status and
run a CUDA matmul. Exit 0 only if torch+CUDA+sm_120 all check out."""

import sys

import torch

print(f"torch          {torch.__version__}")
print(f"cuda available {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("FAIL: no CUDA")
    sys.exit(1)

cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name(0)
print(f"device         {name} (sm_{cap[0]}{cap[1]})")
if cap[0] < 12:
    print(f"FAIL: expected Blackwell sm_120, got sm_{cap[0]}{cap[1]}")
    sys.exit(1)

try:
    import transformers

    print(f"transformers   {transformers.__version__}")
except ImportError:
    print("FAIL: transformers missing")
    sys.exit(1)

a = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
c = a @ b
torch.cuda.synchronize()
assert torch.isfinite(c).all(), "matmul produced non-finite values"
free, total = torch.cuda.mem_get_info()
print(f"matmul bf16    OK")
print(f"vram           {total / 2**30:.1f} GiB total, {free / 2**30:.1f} GiB free")
print("GATE: PASS")
