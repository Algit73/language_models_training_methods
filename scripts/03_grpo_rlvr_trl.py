# %% [markdown]
# # Script 03 - GRPO with a verifiable reward on one GPU (TRL)
#
# Companion doc: `07-rl-for-llms.md`.
# **Status: written against the TRL `main` GRPO docs; NOT yet run on your hardware.** GRPO is the most
# memory- and time-hungry method in this course; treat the first run as a smoke test with tiny settings.
#
# Task: arithmetic word problems with the final answer in `\boxed{}`. The reward is **exact match** plus a
# small format bonus. It is the smallest real verifiable-reward (RLVR) setup.

# %%
import random
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen3-0.6B"
OUT = "runs/grpo-arith"
random.seed(0)


def make(n):
    rows = []
    for _ in range(n):
        a, b, c = random.randint(2, 40), random.randint(2, 40), random.randint(1, 50)
        q = f"Compute {a} * {b} + {c}. Think briefly, then give only the final integer as \\boxed{{answer}}."
        rows.append({"prompt": [{"role": "user", "content": q}], "ground_truth": str(a * b + c)})
    return rows


train = Dataset.from_list(make(2000))

# %% [markdown]
# ## 1. Reward functions
# TRL calls each function with `prompts`, `completions` and every extra dataset column (here `ground_truth`) as
# keyword arguments, and expects one float per completion. Conversational completions arrive as
# `[{"role":"assistant","content":...}]`. Multiple reward functions are summed (or weighted by `reward_weights`).

# %%
BOXED = re.compile(r"\\boxed\{\s*(-?\d+)\s*\}")


def text_of(completion):
    return completion[0]["content"] if isinstance(completion, list) else completion


def correctness(completions, ground_truth, **kwargs):
    out = []
    for c, gt in zip(completions, ground_truth):
        m = BOXED.findall(text_of(c))
        out.append(1.0 if m and m[-1] == gt else 0.0)
    return out


def format_bonus(completions, **kwargs):
    return [0.1 if BOXED.search(text_of(c)) else 0.0 for c in completions]

# %% [markdown]
# ## 2. Configuration that respects 12 GiB
# Memory is dominated by **generation** (KV cache for `num_generations x batch` sequences) plus the backward pass.
# Key constraints from the TRL docs:
# * `per_device_train_batch_size x gradient_accumulation_steps` (x processes) **must be divisible by `num_generations`**.
# * `max_completion_length` is your biggest dial. Start at 128-256.
# * Defaults worth knowing: `loss_type="dapo"`, `beta=0.0` (no reference model, saves memory), `scale_rewards="group"`,
#   `learning_rate=1e-6` (full FT) -> use ~10x for LoRA.
# * `mask_truncated_completions=True` avoids punishing answers that were merely cut off (DAPO recommendation).
# * `use_vllm` is optional and depends on vLLM support for your XPU setup - leave it off until everything else works.

# %%
args = GRPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    num_generations=8,
    max_completion_length=192,
    learning_rate=1e-5,
    loss_type="dapo",
    beta=0.0,
    mask_truncated_completions=True,
    temperature=1.0,
    max_steps=200,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=5,
    log_completions=True,
    report_to="none",
    model_init_kwargs={"dtype": torch.bfloat16},
)
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
trainer = GRPOTrainer(model=MODEL, reward_funcs=[correctness, format_bonus], args=args,
                      train_dataset=train, peft_config=peft_config)

# %%
trainer.train()
trainer.save_model(OUT)

# %% [markdown]
# ## 3. Diagnostics to read (TRL logs them)
# | Metric | Meaning | Action |
# | --- | --- | --- |
# | `reward`, `rewards/correctness/mean` | task progress | should rise; flat from step 0 means base success ~0 -> do SFT first |
# | `frac_reward_zero_std` | share of prompts with identical rewards (no signal) | near 1.0 -> task too hard/easy; filter prompts by difficulty |
# | `completions/mean_length`, `clipped_ratio` | length drift / truncation | growing clipped ratio -> raise `max_completion_length` or add length penalty |
# | `entropy` | exploration | collapsing toward 0 -> reduce LR, add `entropy_coef`, raise temperature |
# | `clip_ratio/*` | fraction of tokens hitting the PPO clip | only meaningful if `num_iterations>1` |
#
# ## Exercises
# 1. Set `num_generations=4` vs `16` (adjust batch sizes so divisibility holds). Compare `frac_reward_zero_std` and wall time.
# 2. Try `loss_type="dr_grpo"` and `scale_rewards="none"`. Does response length behave differently?
# 3. Add a length penalty reward. Check for the classic failure: the model learns to answer without reasoning and accuracy falls on harder problems.
