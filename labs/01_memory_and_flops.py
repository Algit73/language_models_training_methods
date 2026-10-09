# %% [markdown]
# # Lab 01 - What fits on a 12 GiB Arc B580?
#
# A calculator for the three numbers that decide every decision in this course:
# **parameters**, **training memory**, and **compute**. Edit the `CONFIGS` and budget cells and re-run.
#
# Companion doc: `02-hardware-wsl-xpu-setup.md`.
#
# | Setup | Bytes per parameter (rule of thumb) |
# | --- | --- |
# | Full fine-tune, mixed-precision Adam | 16 = 2 (bf16 weight) + 2 (grad) + 4 (fp32 master) + 4 + 4 (Adam m, v) |
# | LoRA on bf16 base | 2 for frozen base + 16 per *trainable adapter* parameter (fp32 weight, grad, Adam m, v) |
# | QLoRA (4-bit base) | about 0.5 + quantization constants for frozen base + 16 per adapter parameter |
#
# Activations, logits and the KV cache come on top.

# %%
from dataclasses import dataclass

GiB = 1024**3


@dataclass
class Cfg:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    inter: int
    vocab: int
    tied: bool


# Example configs written from memory of the public config.json files: CHECK them against the Hub
# before trusting the output. The computed totals below are a built-in sanity check.
CONFIGS = [
    Cfg("Qwen3-0.6B-like", 1024, 28, 16, 8, 128, 3072, 151936, True),
    Cfg("Qwen3-1.7B-like", 2048, 28, 16, 8, 128, 6144, 151936, True),
    Cfg("Qwen3-4B-like", 2560, 36, 32, 8, 128, 9728, 151936, True),
    Cfg("Qwen3-8B-like", 4096, 36, 32, 8, 128, 12288, 151936, False),
]

# %% [markdown]
# ## 1. Parameter count
# Per layer (GQA + SwiGLU):
# $N_{\text{attn}} = d\,(h d_h) + 2\,d\,(g d_h) + (h d_h)\,d$ and $N_{\text{mlp}} = 3\,d\,d_{ff}$,
# plus embeddings $V d$ (and another $V d$ if the output head is not tied).

# %%
def n_params(c: Cfg) -> int:
    attn = c.hidden * c.heads * c.head_dim + 2 * c.hidden * c.kv_heads * c.head_dim + c.heads * c.head_dim * c.hidden
    mlp = 3 * c.hidden * c.inter
    per_layer = attn + mlp + 2 * c.hidden
    emb = c.vocab * c.hidden * (1 if c.tied else 2)
    return emb + c.layers * per_layer + c.hidden


def lora_params(c: Cfg, r: int, attention_only: bool = False) -> int:
    attn = r * (c.hidden + c.heads * c.head_dim) + 2 * r * (c.hidden + c.kv_heads * c.head_dim) + r * (c.heads * c.head_dim + c.hidden)
    mlp = 0 if attention_only else 2 * r * (c.hidden + c.inter) + r * (c.inter + c.hidden)
    return c.layers * (attn + mlp)


for c in CONFIGS:
    print(f"{c.name:18s} total {n_params(c) / 1e9:5.2f} B   LoRA r=16 (all linear) {lora_params(c, 16) / 1e6:6.1f} M")

# %% [markdown]
# ## 2. Training memory estimate
# Activation rule of thumb per layer (Korthikanti et al., with FlashAttention-style attention):
# about $34\,s\,b\,d$ bytes without recomputation; with gradient checkpointing roughly $2\,s\,b\,d$ per layer
# stored plus one layer recomputed. Logits are $b\,s\,V$ values in fp32 *and* their gradient - a hidden
# cost for 150k-token vocabularies (chunked cross-entropy in TRL avoids it).

# %%
def training_gib(c, mode, seq=1024, batch=1, r=16, checkpointing=True):
    n = n_params(c)
    if mode == "full":
        weights_opt = 16 * n
    elif mode == "lora":
        weights_opt = 2 * n + 16 * lora_params(c, r)
    elif mode == "qlora":
        weights_opt = 0.56 * n + 16 * lora_params(c, r)   # 4-bit + ~0.06 B/param for constants (approx.)
    else:
        raise ValueError(mode)
    act_per_layer = (2 if checkpointing else 34) * seq * batch * c.hidden
    act = c.layers * act_per_layer + 34 * seq * batch * c.hidden
    logits = 2 * 4 * batch * seq * c.vocab
    return (weights_opt + act + logits) / GiB


BUDGET_GIB = 12 - 1.5      # leave ~1.5 GiB for runtime, allocator fragmentation, display
print(f"budget: {BUDGET_GIB} GiB usable")
print(f"{'model':18s} {'full FT':>9s} {'LoRA':>9s} {'QLoRA':>9s}   (seq=1024, batch=1, checkpointing)")
for c in CONFIGS:
    row = [training_gib(c, m) for m in ("full", "lora", "qlora")]
    flags = ["ok" if x <= BUDGET_GIB else "NO" for x in row]
    print(f"{c.name:18s} " + " ".join(f"{x:6.1f} {f:>2s}" for x, f in zip(row, flags)))

# %% [markdown]
# ## 3. Inference memory and KV cache
# $\text{KV bytes}=2\cdot L\cdot g\cdot d_h\cdot s\cdot b\cdot \text{bytes}$. For RL you also pay this
# for every concurrently generated sample, so group size $G$ multiplies it.

# %%
def kv_gib(c, seq, batch=1, bytes_per=2):
    return 2 * c.layers * c.kv_heads * c.head_dim * seq * batch * bytes_per / GiB


for c in CONFIGS:
    w4 = 0.56 * n_params(c) / GiB
    print(f"{c.name:18s} 4-bit weights {w4:4.1f} GiB | KV@4k,G=8: {kv_gib(c, 4096, 8):5.2f} GiB | KV@32k,b=1: {kv_gib(c, 32768):5.2f} GiB")

# %% [markdown]
# ## 4. Compute and time
# Training FLOPs $\approx 6ND$ ($N$ parameters, $D$ tokens); inference $\approx 2N$ per token.
# Achieved throughput is peak $\times$ MFU. **You must fill `PEAK_TFLOPS` and `MFU` yourself** (use the
# vendor spec for bf16/XMX and a measured value from `scripts/00_xpu_check.py`); they are assumptions here.

# %%
PEAK_TFLOPS = 100.0     # ASSUMPTION - replace with your measured/datasheet number
MFU = 0.25              # small-batch, single-GPU training rarely exceeds 0.2-0.4


def train_hours(n_params_, tokens, peak=PEAK_TFLOPS, mfu=MFU, lora=False):
    flops = (4 if lora else 6) * n_params_ * tokens   # LoRA skips weight-gradient matmuls of frozen layers (~2/3 of full)
    return flops / (peak * 1e12 * mfu) / 3600


for c in CONFIGS[:3]:
    n = n_params(c)
    print(f"{c.name:18s} 10M-token LoRA SFT ~ {train_hours(n, 10e6, lora=True):5.2f} h | 1B-token CPT (full) ~ {train_hours(n, 1e9):7.1f} h")

# %% [markdown]
# ## 5. Tokens-per-second budget for RL
# RL spends most of its time **generating**. Autoregressive decoding is memory-bandwidth bound:
# $\text{tokens/s} \lesssim \text{bandwidth} / \text{bytes of weights read per token}$.

# %%
BANDWIDTH_GBS = 450.0       # ASSUMPTION - replace with the datasheet number for your card
for c in CONFIGS[:3]:
    wbytes = 2 * n_params(c)
    print(f"{c.name:18s} single-stream decode upper bound ~ {BANDWIDTH_GBS * 1e9 / wbytes:6.0f} tok/s (bf16 weights)")
print("Batching G samples re-uses each weight read, so throughput scales almost linearly until compute-bound.")

# %% [markdown]
# ## Exercises
# 1. Find the largest model for which QLoRA at `seq=2048` fits. Which term dominates: weights, activations, or logits?
# 2. Set `checkpointing=False` and re-run. By how much does it grow?
# 3. Compare LoRA `r=8` vs `r=64` parameter counts. Is the adapter optimizer state ever your bottleneck?
