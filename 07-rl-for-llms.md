# 7. Reinforcement Learning for Language Models

**Goal:** understand policy-gradient RL for LLMs well enough to derive REINFORCE, explain PPO's and GRPO's machinery, choose between the GRPO-family variants, design safe rewards, and run a small RL job on a 12 GB GPU.
**Labs/scripts:** `labs/06_grpo_from_scratch`, `labs/05_rejection_sampling_and_passk`, `scripts/03_grpo_rlvr_trl`.

## 7.1 Setting: the LLM as a policy

* **State** $s_t=(x,y_{<t})$ (prompt + tokens so far); **action** $a_t=y_t$ (next token); **policy** $\pi_\theta(a_t|s_t)$ is the model's softmax; episode = one completion; **reward** $R(x,y)$ usually arrives only at the end (outcome reward).
* Objective: $J(\theta)=\mathbb{E}_{x}\,\mathbb{E}_{y\sim\pi_\theta(\cdot|x)}[R(x,y)]$.
* Differences from classic RL: huge action space (vocabulary), deterministic transitions (you know the next state), strong pretrained initialisation, expensive sampling, sparse outcome rewards.

## 7.2 Policy gradient derivation

Using $\nabla\pi=\pi\nabla\log\pi$ (the *log-derivative trick*):
$$
\nabla_\theta J=\mathbb{E}_{y\sim\pi_\theta}\big[R(x,y)\,\nabla_\theta\log\pi_\theta(y|x)\big],\qquad \log\pi_\theta(y|x)=\sum_t\log\pi_\theta(y_t|s_t)
$$
**REINFORCE** estimates this with samples. The variance is huge, so subtract a **baseline** $b(x)$ that does not depend on the sampled $y$:
$$
\nabla_\theta J=\mathbb{E}\big[(R-b(x))\,\nabla_\theta\log\pi_\theta(y|x)\big]
$$
Unbiased because $\mathbb{E}[b\nabla\log\pi]=b\nabla\sum_y\pi=b\nabla1=0$. The term $A=R-b$ is the **advantage**: positive means "better than usual", raise its probability; negative lowers it.

**Intuition:** RL on outcome rewards increases the probability of whole sampled sequences that scored above the baseline and decreases the others. It only *re-weights what the model can already sample* (chapter 8 discusses what that implies).

## 7.3 The algorithm zoo

### PPO (clipped surrogate)
For token $t$: ratio $\rho_t=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$;
$$
\mathcal{L}_{PPO}=-\mathbb{E}_t\big[\min(\rho_tA_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)\big]
$$
The clip stops one batch from moving the policy too far (you can take several optimiser steps on the same samples). Advantages $A_t$ come from **GAE**: $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$, $A_t=\sum_l(\gamma\lambda)^l\delta_{t+l}$, which needs a **value network** $V$ of roughly the policy's size, trained alongside. Plus a reward model and a reference model: four large networks (policy, value, reward, reference). **Too heavy for 12 GB** beyond tiny models.

### RLOO (REINFORCE leave-one-out)
Sample $G$ completions; baseline for sample $i$ is the mean of the *other* $G-1$ rewards:
$$
A_i=r_i-\frac{1}{G-1}\sum_{j\ne i}r_j
$$
No critic. Unbiased. (Ahmadian et al. 2024, "Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs"; TRL: `RLOOTrainer`.)

### GRPO (Group Relative Policy Optimization)
Introduced in DeepSeekMath (arXiv 2402.03300) and popularised by DeepSeek-R1 (arXiv 2501.12948). For each prompt sample a group $\{o_i\}_{i=1}^G$; compute rewards $r_i$ and
$$
\hat A_i=\frac{r_i-\mathrm{mean}(r)}{\mathrm{std}(r)}
$$
The same advantage is applied to every token of completion $i$. Loss:
$$
\mathcal{L}_{GRPO}=-\frac{1}{\sum_i|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}\Big[\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big)-\beta\,D_{KL}\big[\pi_\theta\Vert\pi_{ref}\big]_{i,t}\Big]
$$
with the k3 KL estimator $D_{KL}=\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1$ (Schulman 2020). With one update per batch, $\rho=1$ in value and only its gradient $\nabla\log\pi_\theta$ matters (lab 06).

**Why it suits one GPU:** no value network; $\beta=0$ removes the reference model too (TRL default now), so memory is policy + optimizer (LoRA) + generation cache.
**Zero-variance groups:** if all $G$ rewards are equal, $\hat A=0$: no learning signal. Healthy training needs a mix of successes and failures per prompt.

### Variants you will see (and the problem each fixes)

| Variant | Problem in original GRPO | Fix | TRL knob (verified in docs) |
| --- | --- | --- | --- |
| **Dr. GRPO** (*Understanding R1-Zero-Like Training*, arXiv 2503.20783) | dividing by $\mathrm{std}$ biases toward very easy/hard questions; dividing each sequence by $\lvert o_i\rvert$ creates a length bias (prefers short correct, long wrong) | no std scaling; normalise by a constant (max length) | `scale_rewards="none"`, `loss_type="dr_grpo"` |
| **DAPO** (arXiv 2503.14476) | sample-level loss under-penalises long bad answers; entropy collapse; wasted zero-variance groups; truncation noise | token-level loss; **clip-higher** ($\epsilon_{high}\approx0.28$); dynamic sampling; mask truncated/overlong completions | `loss_type="dapo"` (now TRL's default), `epsilon_high`, `mask_truncated_completions=True` |
| **GSPO** (arXiv 2507.18071) | token-level importance ratios are noisy for long sequences | sequence-level importance ratio | `importance_sampling_level="sequence"` |
| **Lite PPO** (arXiv 2508.08221) | std computed per small group is noisy | group mean, **batch** std | `scale_rewards="batch"` |
| **Entropy control** (Skywork-OR1, arXiv 2505.22312; "Beyond the 80/20 rule", arXiv 2506.01939) | entropy collapse -> no exploration | entropy bonus (adaptive) / update only high-entropy tokens | `entropy_coef`, `use_adaptive_entropy`, `top_entropy_quantile` |
| **ScaleRL** (arXiv 2510.13786) | instability at scale | fp32 LM head and other recipe details | `cast_lm_head_fp32` |
| **GDPO / multi-reward aggregation** | summing rewards of different scales mixes signals | normalise each reward within group then sum | `multi_objective_aggregation="normalize_then_sum"` |
| **Sampler/trainer mismatch correction** | vLLM vs trainer probabilities differ -> biased gradient | importance-sampling correction (TIS/MIS) | `vllm_importance_sampling_correction` |

Treat these as **ablations, not a menu to enable all at once**. Start with TRL defaults (`dapo`, group-std scaling, `beta=0`), get reward rising, then change one thing.

## 7.4 Rewards: the part that decides success

**Outcome reward from a verifier** (RLVR - RL with verifiable rewards): exact match, unit tests, SQL result equality, JSON schema + fields, environment success. The Tulu 3 post-training recipe (Lambert et al. 2024, arXiv 2411.15124) used this at scale for math/instruction tasks.

**Reward shaping checklist**

1. **Correctness** dominates: 1.0 for verified success.
2. **Format** as a *small* bonus (0.05-0.2) - large format rewards teach format, not substance.
3. **Length/brevity** penalty only after accuracy is good; otherwise the model learns not to think.
4. **Safety penalties** for policy violations (destructive actions, forbidden tools) - larger in magnitude than gains from shortcuts.
5. **Partial credit** (e.g. fraction of test cases passed) can help sparse tasks but invites gaming; compare with binary.
6. **Hidden test sets** for evaluation; never reuse reward tests as your only evaluation.

**Reward hacking - examples to test for:** answer-format regex satisfied without computing; deleting tests; writing to the checker's files; exploiting truncation; repeating a high-reward phrase; producing the ground truth from memory of the benchmark; in GUI agents, exploiting the environment's bugs. Mitigation: read top-reward samples every run, add adversarial unit tests to the reward, sandbox everything, cap steps/time.

**Outcome vs process rewards.** Process reward models (PRMs) score each step (Lightman et al. 2023); they give denser signal but need step labels and can be gamed. For small models, verifiable outcome rewards + careful data filtering are the practical default.

## 7.5 RL on a single 12 GB GPU: the playbook

1. **Start from SFT.** RL needs a non-zero success rate. Check `pass@k` (lab 05): if pass@8 is ~0, do SFT/distillation first. If pass@1 is already ~90%, RL has little to gain.
2. **Model:** 0.5-1.7B first. Use LoRA (r 8-32): Thinking Machines found rank-1 LoRA matches full fine-tuning for policy-gradient RL, because each episode contributes only $O(1)$ bits of information (contrast ~1 bit **per token** in SFT).
3. **Completion length:** 128-512 tokens. Longest sequences cost most (KV cache, quadratic attention). Raise only when needed.
4. **Group size:** $G=4\text{-}8$; larger gives a better baseline and more non-zero-variance groups but costs proportionally more generation.
5. **Batch:** `per_device_batch x grad_accum` must be divisible by `num_generations`.
6. **Learning rate:** LoRA ~1e-5 (full FT default in TRL is 1e-6; LoRA ~10x). Too high -> collapse.
7. **`beta=0`** (no reference model) first; add KL (e.g. 0.001-0.04) only if drift appears.
8. **Sampling temperature 1.0**; RL relies on the policy's own diversity.
9. **Throughput:** generation dominates wall-clock. TRL offers in-process continuous batching (`use_transformers_continuous_batching`) and optional vLLM integration (verify it supports your XPU before depending on it).
10. **Stop rule:** stop when held-out (not training) accuracy plateaus or dev metric degrades.

**Budget estimate.** Each step processes $B\cdot G$ completions of length $L$. With $B=8,G=8,L=200$ that is ~12.8k generated tokens/step. If your decode throughput in batch is ~600 tok/s, each step takes ~20 s plus the update: 200 steps is about an hour or more. Measure with lab 01/script 00 and scale.

## 7.6 Diagnostics (read them every run)

| Signal | Healthy | Unhealthy -> action |
| --- | --- | --- |
| mean reward (train) | rises | flat from step 0 -> SFT first or easier prompts |
| held-out accuracy | rises, then plateaus | falls while train reward rises -> overfit/hacking |
| `frac_reward_zero_std` | < ~0.7 | -> filter prompts by difficulty, raise $G$ or temperature |
| completion length | stable or grows purposefully | explodes (verbosity) or collapses (answering without reasoning) |
| `clipped_ratio` (truncations) | low | high -> raise max length or mask truncated completions |
| entropy | declines slowly | drops to ~0 quickly -> lower LR, add entropy bonus, raise temperature |
| KL to reference (if $\beta>0$) | small, slowly rising | spikes -> lower LR / raise $\beta$ |
| grad norm | steady | spikes -> clip, lower LR |

## 7.7 Does RL add new abilities or sharpen existing ones?

Evidence is mixed and evolving. Several studies report that RLVR mainly **increases pass@1 by concentrating probability on solutions the base model could already sample** ($\text{pass@}k$ at large $k$ may not improve, or may even fall), while longer-horizon recipes and curricula can add skills. For your purposes: measure both pass@1 and $\text{pass@}k$ before and after (lab 05), and expect the largest RL gain when $\text{pass@}k\gg$ pass@1.

## 7.8 Pros and cons of the choices

| Method | Pros | Cons | Use when |
| --- | --- | --- | --- |
| Rejection sampling SFT | simplest, stable, no RL code | only imitates successes; needs non-zero pass@1 | first verifier-based improvement |
| DPO with verifier pairs | offline, stable | needs both success and failure samples; no exploration | you want a cheap preference step |
| RLOO | unbiased, simple, no critic | high variance with small $G$ | small code, moderate compute |
| GRPO (+DAPO/Dr. GRPO) | no critic, well supported, strong on verifiable tasks | sampling cost; zero-variance groups; hacking | main RLVR method on one GPU |
| PPO | per-token credit assignment, flexible | critic memory, tuning complexity | rarely on 12 GB |
| On-policy distillation | dense teacher reward; sample-efficient | needs a good teacher + memory | teacher exists (chapter 5) |

## Checkpoint

* You can derive the policy gradient with baseline and explain why GRPO's group mean is a baseline.
* Lab 06 improves held-out accuracy; you can explain `frac_zero_std` and the effect of $G$.
* You know which knob in TRL implements which published fix, and start from defaults.
* You have a reward-hacking checklist for your task.
