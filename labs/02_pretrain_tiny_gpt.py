# %% [markdown]
# # Lab 02 - Pretraining a tiny GPT (next-token prediction)
#
# Companion doc: `01-foundations-transformer-math.md`. Shared code lives in `toy.py`.
#
# **The task.** The model reads `a1 a2 + b1 b2 =` and must write the sum's digits (least significant
# digit first) followed by a newline. Every answer is machine-checkable, so the same task supports
# SFT, DPO, rejection sampling and RL in later labs.
#
# **The twist.** Our "web corpus" is *noisy*: 50% of the stated answers are random numbers. A model
# trained by plain next-token prediction learns the **mixture** - it can do arithmetic, but when
# *sampled* it only gives the right answer about half the time. This mirrors a real base model:
# capable, but not reliably behaving. Post-training (labs 03-06) fixes that.
#
# Pretraining objective (average over all positions $t$):
# $$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t})$$

# %%
import math
import time

import torch

import toy

torch.manual_seed(0)
dev = toy.device()                      # set LAB_DEVICE=xpu to try the Arc GPU
train_ab, test_ab = toy.split_pairs()   # split by operand pair: test prompts are never trained on
print("train pairs:", len(train_ab), " test pairs:", len(test_ab))

# %% [markdown]
# ## 1. Look at the data
# Decode a few training sequences to see the noise. Remember the answer is written *reversed*.

# %%
gen = torch.Generator().manual_seed(1)
sample = toy.corpus(train_ab[:6], wrong_prob=0.5, gen=gen)
for row, (a, b) in zip(sample, train_ab[:6].tolist()):
    text = "".join(toy.CHARS[i] for i in row.tolist()).replace("\n", "\\n")
    print(f"{a:2d}+{b:2d}={a + b:3d}   corpus line: {text}")

# %% [markdown]
# ## 2. The model
# A 3-layer decoder-only transformer: token+position embeddings, RMSNorm, causal attention, GELU MLP.
# It is ~0.34M parameters - a toy, but architecturally the same family as real LLMs
# (see `00_attention_rope_gqa.py` for the modern variants).

# %%
model = toy.GPT(d=96, layers=3, heads=4).to(dev)
print(f"parameters: {toy.n_params(model):,}")

# %% [markdown]
# ## 3. Train
# Ingredients you will meet again in every real run:
# * **AdamW** with $\beta_2=0.95$ and weight decay,
# * **linear warmup then cosine decay** of the learning rate (`toy.lr_at`),
# * **gradient clipping** at 1.0 (protects against loss spikes),
# * fresh random batches each step (streaming data, single epoch).

# %%
STEPS, BATCH, NOISE = 2000, 200, 0.5


def make_batch(step):
    idx = torch.randint(0, len(train_ab), (BATCH,), generator=gen)
    return toy.corpus(train_ab[idx], NOISE, gen).to(dev)


t0 = time.time()
history = toy.fit(model, make_batch, STEPS, lr=3e-3, loss_from=0, wd=0.01, log_every=400, tag="pretrain")
print(f"trained in {time.time() - t0:.0f}s")

# %% [markdown]
# ## 4. Evaluate: loss vs behaviour
# Two very different numbers:
# * **held-out loss on clean answers** - how well the model *models* correct arithmetic (perplexity),
# * **sampled accuracy** - how often a sampled completion is *actually correct* (what a user sees).
#
# A perfect mixture model has a per-token loss floor of about 0.17 nats here (the first answer digit
# is right with probability $0.5+0.5\cdot0.1=0.55$).

# %%
with torch.no_grad():
    model.eval()
    clean = toy.encode(test_ab, test_ab.sum(1)).to(dev)
    ce = toy.next_token_loss(model, clean, loss_from=toy.P).item()
print(f"clean completion loss: {ce:.3f} nats   perplexity {math.exp(ce):.2f}")
print(f"greedy  accuracy : {toy.accuracy(model, test_ab, greedy=True):.3f}")
print(f"sampled accuracy : {toy.accuracy(model, test_ab, greedy=False, n=5):.3f}   <- this is the number post-training will raise")

# %% [markdown]
# **Reading the result.** Greedy decoding picks the most likely digit and is usually correct, while
# sampling (temperature 1) draws from the noisy mixture and is wrong about as often as the data was.
# Lesson: *a low loss does not mean the model behaves the way you want when sampled*. That gap is
# exactly what SFT, preference optimization and RL close.

# %%
toy.save(model, "base.pt")
print("saved labs/runs/base.pt  (used by labs 03-06)")

# %% [markdown]
# ## Exercises
# 1. Set `NOISE = 0.0`. How do sampled and greedy accuracy change? What does this say about data quality?
# 2. Train with `STEPS = 500`. Is greedy accuracy below sampled accuracy ever possible?
# 3. Double `d` and halve `STEPS`. Which gives a better loss for the same compute? (Chinchilla-style trade-off in miniature.)
