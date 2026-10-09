# %% [markdown]
# # Lab 07 - Distillation: forward vs reverse KL
#
# Companion doc: `05-distillation-and-synthetic-data.md`. Pure PyTorch, runs in seconds, no checkpoint needed.
#
# Distillation trains a small **student** $q_\phi$ to match a **teacher** $p$.
# * **Forward KL** $D_{KL}(p\|q)=\sum_y p(y)\log\frac{p(y)}{q(y)}$ - what ordinary cross-entropy on teacher samples/logits minimises. *Mass-covering*: $q$ must put probability wherever $p$ does.
# * **Reverse KL** $D_{KL}(q\|p)=\sum_y q(y)\log\frac{q(y)}{p(y)}$ - used by on-policy distillation (MiniLLM, GKD variants). *Mode-seeking*: $q$ may ignore modes of $p$ but must not put mass where $p$ has none.
#
# A student that is **too small** to represent the teacher shows the difference clearly. Here the teacher is
# bimodal over 60 bins; the student is a single unimodal bump (a softmax over a quadratic).

# %%
import torch
import torch.nn.functional as F

torch.manual_seed(0)
bins = torch.arange(60).float()


def gaussian_logits(mu, sigma):
    return -0.5 * ((bins - mu) / sigma) ** 2


teacher_logp = torch.logaddexp(gaussian_logits(15, 3) + torch.tensor(0.7).log(),
                               gaussian_logits(42, 3) + torch.tensor(0.3).log())
teacher_logp = teacher_logp - teacher_logp.logsumexp(0)
teacher_p = teacher_logp.exp()
print("teacher mass near mode A (bins 8-22):", teacher_p[8:23].sum().item())
print("teacher mass near mode B (bins 35-49):", teacher_p[35:50].sum().item())

# %% [markdown]
# ## 1. Fit the unimodal student with each divergence

# %%
def fit_student(direction, steps=1500):
    mu = torch.tensor(30.0, requires_grad=True)
    log_sigma = torch.tensor(2.0, requires_grad=True)
    opt = torch.optim.Adam([mu, log_sigma], lr=0.05)
    for _ in range(steps):
        q_logp = F.log_softmax(gaussian_logits(mu, log_sigma.exp()), 0)
        if direction == "forward":
            loss = (teacher_p * (teacher_logp - q_logp)).sum()
        else:
            loss = (q_logp.exp() * (q_logp - teacher_logp)).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return mu.item(), log_sigma.exp().item(), F.log_softmax(gaussian_logits(mu, log_sigma.exp()), 0).detach()


for direction in ("forward", "reverse"):
    mu, sigma, q_logp = fit_student(direction)
    q = q_logp.exp()
    print(f"{direction:8s} KL: mean={mu:5.1f} sigma={sigma:4.1f} | mass on A {q[8:23].sum():.2f}, on B {q[35:50].sum():.2f}, "
          f"in the valley (bins 26-31) {q[26:32].sum():.2f}")

# %% [markdown]
# **Interpretation.**
# * Forward KL places the student *between* the modes (mass in the valley, where the teacher has almost none). For a language model
#   this is the "average of two styles" that is neither - a hallucination-prone blend when sampled.
# * Reverse KL commits to one mode (usually the larger) - sharper, more faithful samples, but it drops the other behaviour (lower diversity).
#
# ## 2. The LLM version
# For token-level distillation at each position: $\mathcal{L}=\sum_t D_{KL}(p_T(\cdot|x_{<t})\,\|\,q_S(\cdot|x_{<t}))$.
# Practical variants:
#
# | Variant | Data | Loss | Notes |
# | --- | --- | --- | --- |
# | Sequence-level KD / "SFT on teacher outputs" | teacher samples | cross-entropy | simplest; works with any API teacher |
# | Logit (white-box) KD | fixed text | forward KL on logits | needs teacher weights and shared tokenizer |
# | On-policy distillation (GKD, MiniLLM) | *student* samples, scored by teacher | reverse KL or JSD | fixes train/inference mismatch (exposure bias) |
#
# Below: temperature $\tau$ softens the teacher so the student also learns the *relative* probabilities of wrong answers
# ("dark knowledge").

# %%
logits = torch.tensor([6.0, 3.0, 1.0, -2.0])
for tau in (1.0, 2.0, 4.0):
    print(f"tau={tau}: teacher distribution {F.softmax(logits / tau, 0).numpy().round(3)}")
print("Remember the gradient scale: multiply the KD loss by tau^2 so it stays comparable to the hard-label loss (Hinton et al., 2015).")

# %% [markdown]
# ## 3. Jensen-Shannon and the generalised family
# $\mathrm{JSD}_\beta=\beta D_{KL}(p\|m)+(1-\beta)D_{KL}(q\|m)$ with $m=\beta p+(1-\beta)q$ interpolates between the two behaviours (GKD, Agarwal et al. 2023).

# %%
def jsd(p_logp, q_logp, beta=0.5):
    p, q = p_logp.exp(), q_logp.exp()
    m_logp = torch.log(beta * p + (1 - beta) * q)
    return beta * (p * (p_logp - m_logp)).sum() + (1 - beta) * (q * (q_logp - m_logp)).sum()


for direction in ("forward", "reverse"):
    _, _, q_logp = fit_student(direction)
    print(f"{direction:8s} student: JSD to teacher = {jsd(teacher_logp, q_logp).item():.4f}")

# %% [markdown]
# ## Exercises
# 1. Change the mixture weights to 0.5/0.5. Which mode does reverse KL choose? (Symmetry breaking depends on the initial `mu`.)
# 2. Let the student be a *mixture of two Gaussians*. Do forward and reverse KL then agree? What does that say about capacity vs objective?
# 3. Why is distilling a reasoning trace from a much stronger teacher risky when the student's tokenizer or context length is smaller?
