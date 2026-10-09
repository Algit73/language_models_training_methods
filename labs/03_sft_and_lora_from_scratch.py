# %% [markdown]
# # Lab 03 - Supervised fine-tuning and LoRA from scratch
#
# Companion doc: `04-sft-and-peft.md`. Needs `runs/base.pt` from lab 02.
#
# **SFT objective** (loss only on the answer tokens, $M$ = completion positions):
# $$\mathcal{L}_{SFT}=-\frac{1}{|M|}\sum_{t\in M}\log p_\theta(x_t\mid x_{<t})$$
# **LoRA**: freeze $W_0$, learn $W=W_0+\frac{\alpha}{r}BA$ with $A\in\mathbb{R}^{r\times d_{in}}$,
# $B\in\mathbb{R}^{d_{out}\times r}$, $B$ initialised to zero so training starts from the base model.
#
# We compare full fine-tuning and LoRA at several ranks on the same clean data and then prove that
# *merging* the adapter changes nothing about the function.

# %%
import copy

import torch

import toy

torch.manual_seed(0)
dev = toy.device()
train_ab, test_ab = toy.split_pairs()
base = toy.load("base.pt")
print("base sampled accuracy:", round(toy.accuracy(base, test_ab, greedy=False, n=5), 3))

# %% [markdown]
# ## 1. Clean SFT data and completion-only loss
# `loss_from=toy.P` masks the prompt: gradients flow only from the 4 answer tokens. In real chat
# data this corresponds to "assistant-only loss" (TRL: `assistant_only_loss=True`).

# %%
def clean_batch(step, n=128):
    idx = torch.randint(0, len(train_ab), (n,))
    ab = train_ab[idx]
    return toy.encode(ab, ab.sum(1)).to(dev)


def run_sft(model, steps=300, lr=3e-4, tag=""):
    toy.fit(model, clean_batch, steps, lr=lr, loss_from=toy.P, warmup=20, tag=tag)
    return toy.accuracy(model, test_ab, greedy=False, n=5)

# %% [markdown]
# ## 2. Full fine-tuning (every weight trains)

# %%
full = copy.deepcopy(base)
acc_full = run_sft(full, lr=3e-4, tag="full")
print(f"full FT   trainable {toy.n_params(full, True):>8,}  sampled acc {acc_full:.3f}")

# %% [markdown]
# ## 3. LoRA at different ranks
# Rule of thumb from the literature: LoRA's best learning rate is ~10x the full-FT one
# (Thinking Machines, "LoRA Without Regret", 2025), so we use 3e-3 here. We adapt *all* linear layers
# (attention and MLP) - attention-only LoRA was found to underperform.

# %%
results = {}
for r in (1, 4, 16):
    m = toy.add_lora(copy.deepcopy(base), r=r, alpha=16).to(dev)
    acc = run_sft(m, lr=3e-3, tag=f"lora r={r}")
    results[r] = (toy.n_params(m, True), acc)
    print(f"LoRA r={r:<2d} trainable {results[r][0]:>8,}  ({100 * results[r][0] / toy.n_params(base):4.1f}% of base)  sampled acc {acc:.3f}")

# %% [markdown]
# ## 4. LoRA starts as the identity and can be merged
# $B=0$ means the adapted model initially equals the base model. After training,
# $W_0 + \frac{\alpha}{r}BA$ is an ordinary matrix, so serving has **zero extra latency**.

# %%
fresh = toy.add_lora(copy.deepcopy(base), r=8, alpha=16).to(dev)
x = toy.prompts(test_ab[:64]).to(dev)
print("fresh LoRA == base? max |logit diff| =", (fresh(x) - base(x)).abs().max().item())

trained = toy.add_lora(copy.deepcopy(base), r=8, alpha=16).to(dev)
run_sft(trained, lr=3e-3, tag="lora r=8")
before = trained(x).detach()
merged = toy.merge_lora(copy.deepcopy(trained))
print("merged == unmerged? max |logit diff| =", (merged(x).detach() - before).abs().max().item())
toy.save(merged, "sft.pt")

# %% [markdown]
# ## 5. What *didn't* change: the adapter is low-rank
# Inspect the singular values of one learned update $\Delta W=\frac{\alpha}{r}BA$. A rank-$r$ LoRA
# can have at most $r$ non-zero singular values: its entire update lives in an $r$-dimensional subspace.

# %%
delta = (trained.blocks[0].up.scale * trained.blocks[0].up.B @ trained.blocks[0].up.A).detach().cpu()
print("top singular values of the first-layer MLP update:", torch.linalg.svdvals(delta)[:10].numpy().round(4))

# %% [markdown]
# ## Reading the results
# * Even rank 1-4 typically recovers most of the gain *for this small, easy dataset*: the information
#   in the data is small compared with adapter capacity. Real datasets with millions of tokens need higher rank.
# * Full FT has ~10x more trainable parameters and needs ~8x more optimizer memory (see lab 01), but no
#   better accuracy here. On a 12 GiB card that decides the choice.
#
# ## Exercises
# 1. Use `targets=("qkv",)` in `toy.add_lora` (attention-only). Compare to all-linear at equal trainable parameters.
# 2. Keep `lr=3e-4` for LoRA. How much slower is it? Why does a zero-initialised $B$ make early steps small?
# 3. Set `loss_from=0` (train on the prompt too). Does accuracy change? What is being wasted?
