# %% [markdown]
# # Lab 06 - GRPO / policy-gradient RL from scratch
#
# Companion doc: `07-rl-for-llms.md`. Needs `runs/base.pt`.
#
# **Policy gradient.** We want to maximise expected reward $J(\theta)=\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}[R(x,y)]$.
# The score-function identity gives $\nabla_\theta J=\mathbb{E}[R\,\nabla_\theta\log\pi_\theta(y|x)]$ (REINFORCE).
# Subtracting a **baseline** $b(x)$ keeps it unbiased and lowers variance:
# $\nabla_\theta J=\mathbb{E}[(R-b)\,\nabla_\theta\log\pi_\theta(y|x)]$.
#
# **GRPO's idea:** sample a *group* of $G$ completions for the same prompt and use the group itself as the baseline:
# $$\hat A_i=\frac{r_i-\operatorname{mean}(r_{1..G})}{\operatorname{std}(r_{1..G})+\varepsilon}$$
# No value network (the memory-hungry part of PPO). Reward = the exact verifier from `toy.reward`.
#
# We implement the three common **loss normalisations** (see TRL `loss_type`): `grpo`, `dr_grpo`, `dapo`.
# In this toy every completion has the same length, so they coincide - read the code to see how they differ
# when lengths vary (real reasoning chains).

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
# ## 1. Advantages from a group

# %%
def group_advantages(rewards, scale="group"):
    # rewards: [B, G]. Zero-variance groups (all right / all wrong) give zero advantage: no signal.
    mean = rewards.mean(1, keepdim=True)
    if scale == "group":
        std = rewards.std(1, keepdim=True)
        return (rewards - mean) / (std + 1e-4)
    return rewards - mean                      # scale="none" (Dr. GRPO recommendation)


r = torch.tensor([[0., 1., 1., 0.], [1., 1., 1., 1.], [0., 0., 0., 0.], [0., 0., 0., 1.]])
print(group_advantages(r))
print("rows 2-3 (no variance) contribute nothing; row 4 (rare success) gets a large positive advantage on the one winner")

# %% [markdown]
# ## 2. The loss
# With one gradient step per batch the importance ratio is exactly 1, but its *gradient* is
# $\nabla\log\pi$; `exp(logp - logp.detach())` reproduces that (same trick TRL uses).
# $$\mathcal{L}=-\frac{1}{Z}\sum_{i=1}^{G}\sum_{t}\; \frac{\pi_\theta(o_{i,t}|\cdot)}{[\pi_\theta(o_{i,t}|\cdot)]_{\text{no grad}}}\hat A_i\;-\;\beta\,\mathrm{KL}$$
# where $Z$ depends on `loss_type`: `grpo` averages per-sequence then over sequences; `dr_grpo`
# divides by a constant $G\cdot L_{max}$; `dapo` divides by the total number of tokens.

# %%
def grpo_loss(policy, seqs, adv, mask, ref_logp=None, beta=0.0, loss_type="dapo", max_len=4):
    logp = toy.completion_logprobs(policy, seqs)               # [N, 4]
    ratio = torch.exp(logp - logp.detach())
    per_tok = -(ratio * adv[:, None])
    if beta > 0 and ref_logp is not None:                      # k3 estimator of KL(pi||ref), Schulman 2020
        d = ref_logp - logp
        per_tok = per_tok + beta * (torch.exp(d) - d - 1)
    per_tok = per_tok * mask
    if loss_type == "grpo":
        return (per_tok.sum(1) / mask.sum(1).clamp(min=1)).mean()
    if loss_type == "dr_grpo":
        return per_tok.sum() / (len(seqs) * max_len)
    return per_tok.sum() / mask.sum().clamp(min=1)             # dapo

# %% [markdown]
# ## 3. Training loop
# Per step: (1) sample $G$ completions for each of $B$ prompts from the *current* policy, (2) score with
# the verifier, (3) compute group advantages, (4) one gradient step. Watch `frac_zero_std`: the share
# of prompts that give no learning signal. It is the health metric for RL on verifiable tasks.

# %%
def train_grpo(steps=120, B=32, G=8, lr=1e-4, beta=0.0, loss_type="dapo", scale="group", seed=0):
    torch.manual_seed(seed)
    pol = copy.deepcopy(base)
    for p in pol.parameters():
        p.requires_grad_(True)
    ref = copy.deepcopy(base).eval() if beta > 0 else None
    opt = torch.optim.AdamW(pol.parameters(), lr=lr, weight_decay=0.0)
    for step in range(steps):
        idx = torch.randint(0, len(train_ab), (B,))
        ab = train_ab[idx].to(dev)
        pr = toy.prompts(ab).repeat_interleave(G, 0)                  # [B*G, P]
        pol.eval()
        seqs = toy.generate(pol, pr)                                  # [B*G, T]
        rew = toy.reward(seqs, ab.repeat_interleave(G, 0)).view(B, G)
        adv = group_advantages(rew, scale).view(-1)
        mask = torch.ones(len(seqs), seqs.shape[1] - toy.P, device=dev)
        with torch.no_grad():
            ref_logp = toy.completion_logprobs(ref, seqs) if ref is not None else None
        pol.train()
        loss = grpo_loss(pol, seqs, adv, mask, ref_logp, beta, loss_type)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0); opt.step()
        if step % 20 == 0 or step == steps - 1:
            zs = ((rew.std(1) == 0).float().mean().item())
            print(f"  step {step:3d} reward {rew.mean():.3f} frac_zero_std {zs:.2f} loss {loss.item():+.4f}")
    return pol


policy = train_grpo()
print("GRPO sampled accuracy on held-out prompts:", round(toy.accuracy(policy, test_ab, greedy=False, n=5), 3))

# %% [markdown]
# ## 4. Does it generalise or just memorise?
# Training prompts and test prompts are different operand pairs. Test accuracy rising confirms the
# policy learned to *sharpen its own arithmetic*, not to memorise answers.

# %%
print("train-prompt accuracy:", round(toy.accuracy(policy, train_ab[:2000], greedy=False, n=3), 3))
print("test-prompt  accuracy:", round(toy.accuracy(policy, test_ab, greedy=False, n=3), 3))
toy.save(policy, "grpo.pt")

# %% [markdown]
# ## 5. Ablations worth running (one at a time)
# * `G=2` vs `G=16`: fewer samples per group means noisier baselines and more zero-variance groups.
# * `scale="none"`: Dr. GRPO argues std-scaling biases toward very easy/very hard questions.
# * `beta=0.04`: add the KL leash (costs a reference forward pass).
# * `lr=1e-3`: observe instability/collapse (entropy falls, reward plateaus or crashes).

# %%
for kwargs in ({"G": 2}, {"G": 16}, {"scale": "none"}):
    pol = train_grpo(steps=80, **kwargs)
    print(f">>> {kwargs}: test accuracy {toy.accuracy(pol, test_ab, greedy=False, n=3):.3f}\n")

# %% [markdown]
# ## Reading the results
# * Rejection sampling (lab 05) only imitates the successes; GRPO also **pushes down** failures
#   (negative advantages), and it uses *fresh* samples each step, so it can keep improving after the
#   initial pass@1 is exhausted - up to what pass@k allows.
# * RL cannot create ability the base lacks: if pass@k is ~0 there is no signal (`frac_zero_std` near 1).
#
# ## Exercises
# 1. Replace the reward with `0.5 * format_ok + 0.5 * correct` (`toy.decode_answer` gives `ok`). Observe partial-credit shaping.
# 2. Introduce reward hacking: reward = `1` if the last token is a newline. What does the policy learn? How would you detect it?
# 3. Implement a second gradient step on the same samples using the clipped PPO ratio $\min(rA,\mathrm{clip}(r,1-\epsilon,1+\epsilon)A)$ (this is TRL's `num_iterations>1`).
