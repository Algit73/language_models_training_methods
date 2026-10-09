# %% [markdown]
# # Script 05 - Merge a LoRA adapter and export
#
# Companion doc: `15-deployment-merging-and-serving.md`.
# **Status: written for the `peft` API; NOT yet run on your hardware.**
#
# A LoRA adapter is a small file that needs its base model at load time. Merging folds the update into the base weights
# ($W\leftarrow W+\frac{\alpha}{r}BA$) so the result is a normal model with zero extra inference cost.

# %%
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "runs/sft-extract"
OUT = "runs/sft-extract-merged"

# %% [markdown]
# ## 1. Merge in bf16 on CPU (safe on small VRAM)
# Merge from a **full-precision** base. Do not merge into a 4-bit-quantized base and save that: you would bake quantization error
# into the weights. (QLoRA practice: train on the 4-bit base, merge into the 16-bit base.)

# %%
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, ADAPTER)
merged = model.merge_and_unload()
merged.save_pretrained(OUT)
AutoTokenizer.from_pretrained(MODEL).save_pretrained(OUT)
print("saved", OUT)

# %% [markdown]
# ## 2. Sanity check: merged == adapter-on-base
# Compare logits on a prompt; differences should be at bf16 rounding level.

# %%
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok("Extract name, city: Ava goes to Oslo.", return_tensors="pt").input_ids
with torch.no_grad():
    a = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16), ADAPTER)(ids).logits
    b = AutoModelForCausalLM.from_pretrained(OUT, dtype=torch.bfloat16)(ids).logits
print("max |logit diff|:", (a - b).abs().max().item())

# %% [markdown]
# ## 3. Quantised formats (next steps; check each tool's current docs)
# * **GGUF** for llama.cpp: convert the merged HF directory with llama.cpp's conversion script, then quantise (e.g. Q4_K_M / Q5_K_M / Q8_0).
#   llama.cpp has SYCL and Vulkan backends that can use Intel GPUs - verify current support for Battlemage and WSL2.
# * **AWQ / GPTQ**: post-training weight quantisation with calibration data; needs runtime kernels for your backend.
# * **bitsandbytes** NF4/INT8 is for *training/inference inside PyTorch*, and now has an Intel XPU backend.
#
# After any quantisation, **re-run your evaluation** (script 04): quantisation changes behaviour, especially for small models
# and long reasoning chains.
