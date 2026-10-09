# %% [markdown]
# # Lab 05 - Verifier-based rejection sampling (RFT) and pass@k
#
# Companion docs: `08-reasoning-models.md`, `14-evaluation.md`. Needs `runs/base.pt`.
#
# **Recipe (a.k.a. RFT / STaR-style / "best-of-N distillation")**
# 1. sample $k$ completions per prompt from the current model,
# 2. keep only those a *verifier* accepts,
# 3. fine-tune on the kept samples with ordinary SFT,
# 4. optionally repeat (expert iteration).
#
# It is the simplest way to turn a verifier into a training signal - no RL algorithm needed.
#
# **pass@k.** If you draw $n$ samples and $c$ are correct, the unbiased estimate of "at least one of $k$ is correct" is
# $$\text{pass@}k = 1-\binom{n-c}{k}\big/\binom{n}{k}$$
# (Chen et al., 2021, the Codex paper). It tells you the *headroom* for rejection sampling and RL:
# if pass@8 is high but pass@1 is low, the model *can* solve it and only needs sharpening.

# %%
import copy

import torch

import toy

torch.manual_seed(0)
dev = toy.device()
train_ab, test_ab = toy.split_pairs()
base = toy.load("base.pt")

# %% [markdown]
# ## 1. Measure the headroom first

# %%
N = 16
ab = test_ab[:1000].to(dev)
pr = toy.prompts(ab)
correct = torch.stack([toy.reward(toy.generate(base, pr), ab) for _ in range(N)]).sum(0)   # c per prompt
for k in (1, 2, 4, 8, 16):
    est = sum(toy.pass_at_k(N, int(c), k) for c in correct.tolist()) / len(correct)
    print(f"base pass@{k:<2d} = {est:.3f}")

# %% [markdown]
# ## 2. Sample, verify, keep

# %%
def collect(model, ab_all, k=8, per_chunk=500):
    kept = []
    for i in range(0, len(ab_all), per_chunk):
        chunk = ab_all[i:i + per_chunk].to(dev)
        pr = toy.prompts(chunk)
        seen = set()
        for _ in range(k):
            s = toy.generate(model, pr)
            ok = toy.reward(s, chunk).bool()
            for j in torch.nonzero(ok).flatten().tolist():
                key = (i + j, tuple(s[j].tolist()))
                if key not in seen:                # de-duplicate identical answers per prompt
                    seen.add(key); kept.append(s[j])
    return torch.stack(kept)


kept = collect(base, train_ab[:4000])
print("kept", len(kept), "verified samples out of", 4000 * 8, "drawn")

# %% [markdown]
# ## 3. SFT on the verified samples (completion-only loss)

# %%
def sft_on(model, data, steps=250, lr=3e-4):
    def batch(step):
        idx = torch.randint(0, len(data), (128,))
        return data[idx].to(dev)
    toy.fit(model, batch, steps, lr=lr, loss_from=toy.P, warmup=20)
    return model


student = sft_on(copy.deepcopy(base), kept)
print("after 1 round of RFT: sampled accuracy", round(toy.accuracy(student, test_ab, greedy=False, n=5), 3))

# %% [markdown]
# ## 4. Iterate (expert iteration)
# Re-sample from the *improved* model: its samples are more often correct, so the next dataset is
# bigger and on-distribution. Gains shrink each round as pass@k saturates.

# %%
current = student
for rnd in (2, 3):
    data = collect(current, train_ab[:4000])
    current = sft_on(current, data, steps=150, lr=2e-4)
    print(f"round {rnd}: kept {len(data)}  sampled accuracy {toy.accuracy(current, test_ab, greedy=False, n=5):.3f}")
toy.save(current, "rft.pt")

# %% [markdown]
# ## 5. A broken verifier teaches a broken model
# Replace the verifier with one that only checks *format* (three digits then newline). Everything that
# "looks like an answer" is accepted, so the model learns to look right, not to be right.

# %%
def weak_reward(seqs, ab):
    _, ok = toy.decode_answer(seqs)
    return ok.float()


_orig = toy.reward
toy.reward = weak_reward
kept_bad = collect(base, train_ab[:4000])
toy.reward = _orig
bad = sft_on(copy.deepcopy(base), kept_bad)
print("format-only verifier ->", round(toy.accuracy(bad, test_ab, greedy=False, n=5), 3), "(compare with the real verifier above)")

# %% [markdown]
# ## Exercises
# 1. Plot kept-samples vs round. Why does the dataset grow even though it is the same prompt set?
# 2. Use $k=2$ instead of 8 in the first round. How much headroom do you lose? Relate to pass@k.
# 3. Keep only the *shortest* correct answer per prompt. In a real reasoning task this trades accuracy for brevity - what could go wrong?
