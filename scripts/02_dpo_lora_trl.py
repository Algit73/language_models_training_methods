# %% [markdown]
# # Script 02 - DPO with LoRA (TRL)
#
# Companion doc: `06-preference-optimization.md`.
# **Status: written against the TRL `main` docs (DPOTrainer/DPOConfig); NOT yet run on your hardware.**
#
# Start DPO from an **SFT'd** model (merge or continue from the adapter of script 01). With PEFT, TRL uses the
# base model with the adapter disabled as the reference policy, so you do not pay for a second copy of the weights.

# %%
import torch
from datasets import Dataset
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

MODEL = "Qwen/Qwen3-0.6B"     # replace with your merged SFT checkpoint directory
OUT = "runs/dpo"

# %% [markdown]
# ## 1. Preference data
# Format (conversational, explicit prompt - the recommended layout):
# `{"prompt": [...], "chosen": [...], "rejected": [...]}`.
# Build pairs with a *verifier* whenever you can (correct vs incorrect, valid JSON vs invalid JSON), otherwise from
# human edits (edited = chosen, original = rejected) or a judge model with positions randomised.

# %%
def pair(question, good, bad):
    return {"prompt": [{"role": "user", "content": question}],
            "chosen": [{"role": "assistant", "content": good}],
            "rejected": [{"role": "assistant", "content": bad}]}


rows = [pair("Return JSON with key 'ok' set to true.", '{"ok": true}', "Sure! ok is true.") for _ in range(200)]
train = Dataset.from_list(rows)

# %% [markdown]
# ## 2. Configuration
# * `beta=0.1` (default): the KL-leash coefficient. Lower = move further from the reference.
# * learning rate: the TRL docs note ~1e-5 is typical with adapters (default is 1e-6 for full fine-tuning).
# * `loss_type`: `["sigmoid"]` is standard DPO. TRL also exposes `ipo`, `hinge`, `sigmoid_norm` (SimPO-style length
#   normalisation), `bco_pair`, `apo_zero`, `sft`, and weighted combinations such as `["sigmoid","sft"]`.
# * `precompute_ref_log_probs=True` is an option when you keep an explicit reference model and need its memory back.

# %%
args = DPOConfig(
    output_dir=OUT,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    beta=0.1,
    loss_type=["sigmoid"],
    max_length=512,
    num_train_epochs=1,
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    report_to="none",
    model_init_kwargs={"dtype": torch.bfloat16},
)
peft_config = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
trainer = DPOTrainer(model=MODEL, args=args, train_dataset=train, peft_config=peft_config)

# %%
trainer.train()
trainer.save_model(OUT)

# %% [markdown]
# ## 3. What to watch in the logs
# | Metric | Healthy | Warning |
# | --- | --- | --- |
# | `rewards/margins` | grows steadily | explodes within a few steps -> LR too high / pairs too easy |
# | `rewards/accuracies` | rises toward ~0.7-1.0 | stuck at 0.5 -> LR too low or pairs uninformative |
# | `logps/chosen` | roughly flat | drops a lot -> likelihood displacement; add an SFT term or lower LR |
# | response length | stable | grows -> length exploitation; try `sigmoid_norm` or add length control |
#
# Always re-measure the **task metric** after DPO: a rising reward margin alone proves nothing.
