# %% [markdown]
# # Script 04 - Generate, score, compare (base vs adapter) with confidence intervals
#
# Companion doc: `14-evaluation.md`.
# **Status: written for the `transformers` + `peft` API; NOT yet run on your hardware.**
#
# Evaluating by *behaviour* on held-out prompts with a deterministic checker is what tells you whether training worked.

# %%
import json
import math
import random

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-0.6B"
ADAPTER = "runs/sft-extract"          # output of script 01
DEVICE = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
random.seed(123)

# %% [markdown]
# ## 1. A held-out set with an exact checker
# Generate fresh data (different seed from training) so it is not memorised. In real projects freeze this set *before* training.

# %%
NAMES = ["Ava", "Liam", "Noor", "Mia", "Omar", "Zoe"]
CITIES = ["Oslo", "Lima", "Kyoto", "Cairo", "Quito", "Perth"]


def make_eval(n=100):
    items = []
    for _ in range(n):
        name, city, day, nights = random.choice(NAMES), random.choice(CITIES), random.randint(1, 28), random.randint(2, 9)
        text = f"{name} will travel to {city} on the {day}th and stay for {nights} nights."
        items.append({"text": text, "gold": {"name": name, "city": city, "day": day, "nights": nights}})
    return items


def check(output: str, gold: dict) -> dict:
    try:
        obj = json.loads(output.strip())
    except Exception:
        return {"valid": 0, "exact": 0}
    return {"valid": 1, "exact": int(obj == gold)}

# %% [markdown]
# ## 2. Greedy generation helper (deterministic; report decoding settings with results)

# %%
tok = AutoTokenizer.from_pretrained(MODEL)


@torch.no_grad()
def answer(model, text, max_new_tokens=80):
    msgs = [{"role": "user", "content": f"Extract name, city, day, nights as JSON.\nText: {text}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEVICE)
    out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def evaluate(model, items):
    res = [check(answer(model, it["text"]), it["gold"]) for it in items]
    return [r["exact"] for r in res], [r["valid"] for r in res]

# %% [markdown]
# ## 3. Base vs adapter, paired (same items)

# %%
items = make_eval(100)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
base_exact, base_valid = evaluate(base, items)

tuned = PeftModel.from_pretrained(base, ADAPTER).eval()
tuned_exact, tuned_valid = evaluate(tuned, items)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


for name, ex, va in (("base", base_exact, base_valid), ("adapter", tuned_exact, tuned_valid)):
    lo, hi = wilson(sum(ex), len(ex))
    print(f"{name:8s} exact {sum(ex) / len(ex):.3f} [{lo:.3f},{hi:.3f}]   valid-JSON {sum(va) / len(va):.3f}")

# %% [markdown]
# ## 4. Inspect failures (always)
# A number hides *why* a model fails. Read at least 20 failures and label them: wrong value, invalid format, refusal, hallucinated field.

# %%
fails = [i for i, e in enumerate(tuned_exact) if not e][:20]
for i in fails:
    print("TEXT:", items[i]["text"], "\nGOLD:", items[i]["gold"], "\nOUT :", answer(tuned, items[i]["text"]), "\n---")

# %% [markdown]
# Use the paired bootstrap / McNemar code from `labs/08_evaluation_statistics.py` to decide whether the difference is real.
