# %% [markdown]
# # Script 00 - Is the Arc B580 visible, and how fast is it?
#
# Companion doc: `02-hardware-wsl-xpu-setup.md`.
# **Status: written from the PyTorch XPU docs, not yet run on your machine.** Run it inside WSL2 Ubuntu 24.04
# after `pip install torch --index-url https://download.pytorch.org/whl/xpu`.
#
# This is the gate before any training: if `torch.xpu.is_available()` is False, fix the driver/WSL setup first.

# %%
import platform
import time

import torch

print("python :", platform.python_version(), "|", platform.platform())
print("torch  :", torch.__version__)
has_xpu = hasattr(torch, "xpu") and torch.xpu.is_available()
print("XPU available:", has_xpu)
if not has_xpu:
    raise SystemExit("No XPU. Check: Windows Intel driver, `wsl --update`, Ubuntu 24.04 (WSL2), torch XPU wheel.")

# %%
props = torch.xpu.get_device_properties(0)
print("device :", props.name)
print("memory : %.1f GiB total" % (props.total_memory / 1024**3))
print("bf16 supported:", torch.xpu.is_bf16_supported() if hasattr(torch.xpu, "is_bf16_supported") else "unknown")

# %% [markdown]
# ## Measured matmul throughput
# Use the **measured** number as `PEAK_TFLOPS` in lab 01 instead of a datasheet value. A large square matmul is
# close to the best case; real training reaches a fraction of it (that fraction is the MFU).

# %%
def bench(dtype, n=4096, iters=20):
    a = torch.randn(n, n, device="xpu", dtype=dtype)
    b = torch.randn(n, n, device="xpu", dtype=dtype)
    for _ in range(3):
        a @ b
    torch.xpu.synchronize()
    t0 = time.time()
    for _ in range(iters):
        a @ b
    torch.xpu.synchronize()
    dt = (time.time() - t0) / iters
    return 2 * n**3 / dt / 1e12


for dt in (torch.float32, torch.bfloat16, torch.float16):
    print(f"{str(dt):16s} {bench(dt):7.1f} TFLOPS (measured, 4096x4096 matmul)")

# %% [markdown]
# ## Memory and bandwidth probes
# * Allocate until failure to learn the *usable* VRAM (less than the nominal 12 GiB).
# * Time a large device-to-device copy to estimate memory bandwidth (feeds the decode-speed bound in lab 01).

# %%
x = torch.empty(512 * 1024**2, dtype=torch.uint8, device="xpu")     # 512 MiB
y = torch.empty_like(x)
torch.xpu.synchronize()
t0 = time.time()
for _ in range(20):
    y.copy_(x)
torch.xpu.synchronize()
dt = (time.time() - t0) / 20
print(f"copy bandwidth ~ {2 * x.numel() / dt / 1e9:.0f} GB/s (read+write)")

chunks, gib = [], 0
try:
    while True:
        chunks.append(torch.empty(1024**3, dtype=torch.uint8, device="xpu"))
        gib += 1
except RuntimeError:
    pass
print(f"allocated {gib} GiB before failing (usable VRAM for one process)")
