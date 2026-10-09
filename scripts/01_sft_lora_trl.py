# %% [markdown]
# # Script 01 - LoRA SFT of a small chat model with TRL (Arc B580, WSL2)
#
# Companion doc: `04-sft-and-peft.md`.
# **Status: written against the TRL `main` docs (SFTTrainer/SFTConfig); NOT yet run on your hardware.**
# Expected install inside WSL2 (versions move quickly - prefer the newest that imports cleanly):
# `pip install torch --index-url https://download.pytorch.org/whl/xpu` then
# `pip install transformers datasets peft trl accelerate`.
#
# Task: JSON extraction (text -> JSON). It has a deterministic metric (schema-valid + field match) so
# you can prove the adapter helps. Replace `make_examples` with your own data.

# %%
import json
import random

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL = "Qwen/Qwen3-0.6B"      # smoke-test size (it is a documented TRL example model). Move to 1.7B/4B once this works.
OUT = "runs/sft-extract"
random.seed(0)

# %% [markdown]
# ## 1. Data: prompt/completion in conversational format
# With a *prompt-completion* dataset TRL computes the loss on the completion only by default
# (`completion_only_loss=None` -> completion for prompt-completion data). For full chat transcripts use
# `assistant_only_loss=True`, which needs a chat template with `{% generation %}` markers (TRL patches some families automatically).

# %%
NAMES = ["Ava", "Liam", "Noor", "Mia", "Omar", "Zoe"]
CITIES = ["Oslo", "Lima", "Kyoto", "Cairo", "Quito", "Perth"]


def make_examples(n):
    rows = []
    for _ in range(n):
        name, city, day = random.choice(NAMES), random.choice(CITIES), random.randint(1, 28)
        text = f"{name} will travel to {city} on the {day}th and stay for {random.randint(2, 9)} nights."
        nights = int(text.split("stay for ")[1].split()[0])
        target = {"name": name, "city": city, "day": day, "nights": nights}
        rows.append({
            "prompt": [{"role": "user", "content": f"Extract name, city, day, nights as JSON.\nText: {text}"}],
            "completion": [{"role": "assistant", "content": json.dumps(target)}],
        })
    return rows


train = Dataset.from_list(make_examples(800))
evalset = Dataset.from_list(make_examples(100))

# %% [markdown]
# ## 2. LoRA config
# From "LoRA Without Regret": apply LoRA to **all linear layers** (attention *and* MLP) and use a learning rate
# roughly **10x** what you would use for full fine-tuning. `r=16, alpha=32` is a fine start; with the `alpha/r`
# scaling the best LR is nearly independent of rank.

# %%
peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")

# %% [markdown]
# ## 3. Training config (sized for 12 GiB)
# * `per_device_train_batch_size=1` + `gradient_accumulation_steps=16`: same math as batch 16, a fraction of the activation memory.
# * `gradient_checkpointing` (TRL default True): recompute activations in backward, trading ~30% time for a lot of memory.
# * `max_length` is the main memory dial; logits cost `batch x length x vocab x 4 bytes`.
# * `bf16=True`: the B-series supports bf16. (Enable `GradScaler`-style fp16 only if you must: it needs FP64 on some GPUs.)

# %%
args = SFTConfig(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=2e-4,
    num_train_epochs=1,
    max_length=512,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    report_to="none",
    model_init_kwargs={"dtype": torch.bfloat16},
)
trainer = SFTTrainer(model=MODEL, args=args, train_dataset=train, eval_dataset=evalset, peft_config=peft_config)
print("device in use:", trainer.model.device)          # must say xpu:0, not cpu
trainer.model.print_trainable_parameters()

# %%
trainer.train()
trainer.save_model(OUT)                                 # saves the adapter only (tens of MB)

# %% [markdown]
# ## 4. Evaluate *behaviour*, not just loss
# Compare base vs adapter on held-out prompts with the deterministic metric. See `04_generate_and_score.py`.
#
# ## Things to record (copy into your experiment table)
# `transformers/trl/peft/torch` versions, Windows driver version, peak memory (`torch.xpu.max_memory_allocated()`),
# tokens/s, training loss curve, eval metric with 95% CI (lab 08).
#
# ## Exercises
# 1. Change `target_modules` to `["q_proj","v_proj"]` and compare at the same learning rate and trainable-parameter budget.
# 2. Raise `max_length` until you run out of memory; record the failing value and peak allocation.
# 3. Turn off `gradient_checkpointing`. How do memory and step time change?
