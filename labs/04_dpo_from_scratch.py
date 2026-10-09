# %% [markdown]
# # Lab 04 - DPO from scratch
#
# Companion doc: `06-preference-optimization.md`. Needs `runs/base.pt`.
#
# Given a prompt $x$, a preferred completion $y_w$ and a rejected one $y_l$:
# $$\mathcal{L}_{DPO}=-\log\sigma\!\Big(\beta\big[\underbrace{\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}}_{\hat r_w}-\underbrace{\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}}_{\hat r_l}\big]\Big)$$
# $\hat r=\beta\log(\pi_\theta/\pi_{ref})$ is the *implicit reward*. No reward model, no sampling loop
# during training - but the pairs must come from somewhere. Here the verifier builds them:
# **chosen = a correct sample, rejected = an incorrect sample**, both drawn from the base model.

# %%
import copy

import torch
import torch.nn.functional as F

import toy

torch.manual_seed(0)
dev = toy.device()
train_ab, test_ab = toy.split_pairs()
policy = toy.load("base.pt")
reference = copy.deepcopy(policy).eval()
for p in reference.parameters():
    p.requires_grad_(False)
print("base sampled accuracy:", round(toy.accuracy(policy, test_ab, greedy=False, n=5), 3))

# %% [markdown]
# ## 1. Build preference pairs with a verifier
# For each prompt sample $k$ completions from the reference. Keep prompts where at least one is correct
# and one is wrong. This is **on-policy** data (samples come from the model being trained), which
# tends to work better than off-policy pairs written by a different model.

# %%
def build_pairs(ab, k=8, max_pairs=4000):
    chosen, rejected, prompt_rows = [], [], []
    for i in range(0, len(ab), 500):
        chunk = ab[i:i + 500].to(dev)
        pr = toy.prompts(chunk)
        samples = [toy.generate(reference, pr) for _ in range(k)]              # k x [n, T]
        rewards = torch.stack([toy.reward(s, chunk) for s in samples])          # [k, n]
        for j in range(len(chunk)):
            good = [s[j] for s, r in zip(samples, rewards[:, j]) if r == 1]
            bad = [s[j] for s, r in zip(samples, rewards[:, j]) if r == 0]
            if good and bad:
                chosen.append(good[0]); rejected.append(bad[0]); prompt_rows.append(chunk[j])
        if len(chosen) >= max_pairs:
            break
    return torch.stack(chosen), torch.stack(rejected)


chosen, rejected = build_pairs(train_ab)
print("preference pairs:", len(chosen))
print("example chosen  :", repr("".join(toy.CHARS[i] for i in chosen[0].tolist())))
print("example rejected:", repr("".join(toy.CHARS[i] for i in rejected[0].tolist())))

# %% [markdown]
# ## 2. The DPO loss
# `completion_logprobs` returns per-token log-probabilities of the 4 answer tokens; summing gives
# $\log\pi(y|x)$. The reference log-probs are constants, so we **precompute** them once (TRL offers
# the same trick: `precompute_ref_log_probs=True`) - this is how DPO saves memory on one GPU.

# %%
def seq_logp(model, seqs):
    return toy.completion_logprobs(model, seqs).sum(-1)


with torch.no_grad():
    ref_c, ref_r = seq_logp(reference, chosen.to(dev)), seq_logp(reference, rejected.to(dev))


def dpo_loss(policy, c, r, ref_c_, ref_r_, beta):
    rw = beta * (seq_logp(policy, c) - ref_c_)            # implicit reward of chosen
    rl = beta * (seq_logp(policy, r) - ref_r_)            # implicit reward of rejected
    loss = -F.logsigmoid(rw - rl).mean()
    return loss, rw.detach(), rl.detach()

# %% [markdown]
# ## 3. Train and watch the metrics TRL would log
# * `margin = r_w - r_l` should grow, `accuracy = P(r_w > r_l)` should approach 1.
# * Watch **`logp(chosen)`**: DPO can lower the likelihood of *both* responses as long as the gap widens
#   ("likelihood displacement"). If chosen log-prob falls a lot, quality can degrade - a known failure mode.

# %%
def train_dpo(beta=0.1, steps=200, lr=3e-4, bs=64, seed=0):
    torch.manual_seed(seed)
    pol = copy.deepcopy(reference)
    for p in pol.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        pol.train()
        idx = torch.randint(0, len(chosen), (bs,))
        loss, rw, rl = dpo_loss(pol, chosen[idx].to(dev), rejected[idx].to(dev), ref_c[idx], ref_r[idx], beta)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if step % 50 == 0 or step == steps - 1:
            with torch.no_grad():
                lc = seq_logp(pol, chosen[idx].to(dev)).mean().item()
            print(f"  beta={beta} step {step:3d} loss {loss.item():.3f} margin {(rw - rl).mean().item():+.3f} "
                  f"acc {(rw > rl).float().mean().item():.2f} logp(chosen) {lc:+.2f}")
    return pol


policy = train_dpo(beta=0.1)
print("DPO (beta=0.1) sampled accuracy:", round(toy.accuracy(policy, test_ab, greedy=False, n=5), 3))

# %% [markdown]
# ## 4. The role of $\beta$
# $\beta$ is the inverse strength of the KL leash to the reference: small $\beta$ lets the policy move
# far (faster gains, more risk of drift); large $\beta$ stays close. Compare:

# %%
for beta in (0.02, 0.1, 0.5):
    pol = train_dpo(beta=beta, steps=150)
    print(f">>> beta={beta}: sampled accuracy {toy.accuracy(pol, test_ab, greedy=False, n=5):.3f}\n")

# %% [markdown]
# ## Reading the results
# * Pairs came from a *verifier*, so DPO here is "offline RL with a perfect reward". It improves
#   sampled accuracy without any rollout during training.
# * It cannot discover answers the reference never samples: if a prompt has no correct sample it yields no pair.
#   (That is what online RL - lab 06 - can do.)
#
# ## Exercises
# 1. Use `k=2` when building pairs. How many pairs do you get and does accuracy still improve?
# 2. Add an SFT term on chosen answers: `loss = dpo + 0.1 * (-seq_logp(pol, c).mean())`. Does logp(chosen) still fall? (This is the idea behind RPO / `loss_type=["sigmoid","sft"]` in TRL.)
# 3. Make the rejected answers trivial (random digits instead of model samples). Does DPO still help? Why are *plausible* negatives better?
