# 6. Preference Optimization: from RLHF to DPO and Friends

**Goal:** understand how "which answer is better" becomes a training signal, derive DPO yourself, and know when each of DPO/IPO/KTO/ORPO/SimPO is appropriate.
**Labs/scripts:** `labs/04_dpo_from_scratch`, `labs/11_reward_model_and_infonce`, `scripts/02_dpo_lora_trl`.

## 6.1 Why preference learning

Demonstrations (SFT) tell the model *what to say*. Many qualities - helpfulness, conciseness, groundedness, safety, correct format when alternatives are plausible - are easier to **compare** than to write. A pair "A is better than B" also penalises specific bad behaviour, which SFT cannot do.

## 6.2 Classic RLHF (three stages)

1. SFT model $\pi_{ref}$.
2. **Reward model** $r_\phi(x,y)$ trained on comparisons.
3. **RL** (PPO) maximising the reward with a KL leash to $\pi_{ref}$.

**Bradley-Terry reward model.** Assume $P(y_w\succ y_l\mid x)=\sigma\big(r(x,y_w)-r(x,y_l)\big)$. Fit by maximum likelihood:
$$
\mathcal{L}_{RM}(\phi)=-\mathbb{E}\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)
$$
(Lab 11 recovers a hidden reward from noisy comparisons; real annotators agree only ~70-80% of the time, so the ceiling on accuracy is well below 100%.)

**KL-regularised RL objective.**
$$
\max_{\pi}\ \mathbb{E}_{x,\,y\sim\pi}\big[r(x,y)\big]-\beta\,D_{KL}\big(\pi(\cdot|x)\,\Vert\,\pi_{ref}(\cdot|x)\big)
$$
The KL term keeps the policy where the reward model is trustworthy; without it the policy finds exploits (over-optimisation, Goodhart - lab 11 section 2).

## 6.3 Deriving DPO

1. The KL-regularised objective has the closed-form optimum
$$
\pi^*(y|x)=\frac{1}{Z(x)}\,\pi_{ref}(y|x)\,\exp\!\big(r(x,y)/\beta\big),\quad Z(x)=\sum_y\pi_{ref}(y|x)e^{r(x,y)/\beta}
$$
2. Invert: $r(x,y)=\beta\log\dfrac{\pi^*(y|x)}{\pi_{ref}(y|x)}+\beta\log Z(x)$.
3. Plug into Bradley-Terry. $Z(x)$ depends only on $x$, so it **cancels** in the difference $r(x,y_w)-r(x,y_l)$.
4. Replace $\pi^*$ by the trainable policy $\pi_\theta$:
$$
\boxed{\mathcal{L}_{DPO}=-\mathbb{E}_{(x,y_w,y_l)}\log\sigma\!\Big(\beta\Big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-\log\tfrac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\Big]\Big)}
$$

The policy *is* the reward model: $\hat r_\theta=\beta\log(\pi_\theta/\pi_{ref})$ ("Your language model is secretly a reward model", Rafailov et al. 2023, arXiv 2305.18290).

**Gradient** (why it behaves sensibly):
$$
\nabla_\theta\mathcal{L}=-\beta\,\mathbb{E}\Big[\underbrace{\sigma(\hat r_l-\hat r_w)}_{\text{weight: larger when the implicit ranking is wrong}}\big(\nabla\log\pi_\theta(y_w|x)-\nabla\log\pi_\theta(y_l|x)\big)\Big]
$$
Examples the model already ranks correctly contribute little; mis-ranked pairs dominate.

**Sequence log-probabilities** are sums of token log-probs: $\log\pi(y|x)=\sum_t\log\pi(y_t|x,y_{<t})$. Hence **longer responses have larger magnitude log-ratios** - the source of DPO's length sensitivity.

**Memory on one GPU.** You need $\pi_\theta$ and $\pi_{ref}$. With LoRA, $\pi_{ref}$ is the same network with the adapter disabled (no second copy). Or precompute reference log-probs once (`precompute_ref_log_probs=True`). Sequences are processed in pairs (chosen and rejected), so activation memory is about $2\times$ SFT.

## 6.4 What $\beta$ does

$\beta$ multiplies the log-ratio. Small $\beta$: weak leash, policy drifts far (large gains *and* risks). Large $\beta$: stays near reference. Default $0.1$; try $\{0.02,0.1,0.5\}$ as in lab 04.

## 6.5 The family

| Method | Change vs DPO | Data | Reference model? | Use when |
| --- | --- | --- | --- | --- |
| **DPO** | Bradley-Terry on implicit reward | pairs | yes (or adapter-off) | default |
| **IPO** (Azar et al. 2023; arXiv 2310.12036) | squared loss pulling the margin to $1/(2\tau)$ instead of sigmoid | pairs | yes | preferences near-deterministic; DPO overfits/drives margins to infinity |
| **KTO** (Ethayarajh et al. 2024) | prospect-theory loss on single responses labelled good/bad | unpaired binary labels | yes | you only have thumbs-up/down |
| **ORPO** (Hong et al. 2024) | SFT loss + odds-ratio penalty, single stage | pairs | **no** | save memory/compute; combine SFT and preference |
| **SimPO** (Meng et al. 2024; arXiv 2405.14734) | reward = length-normalised mean log-prob, margin $\gamma$: $-\log\sigma\big(\tfrac{\beta}{\lvert y_w\rvert}\log\pi(y_w)-\tfrac{\beta}{\lvert y_l\rvert}\log\pi(y_l)-\gamma\big)$ | pairs | **no** | length bias; memory limited |
| **Online / on-policy DPO** | pairs re-sampled from current policy and labelled by judge/verifier | prompts + labeller | yes | distribution shift; you have a labeller |
| **RLHF-PPO / GRPO with RM** | explicit reward model + RL | prompts + RM | yes | when preference is hard to capture in pairs; heavier |

TRL: stable `DPOTrainer` (with `loss_type` options such as `sigmoid`, `ipo`, `hinge`, `sigmoid_norm` (SimPO-style length normalisation), `bco_pair`, `apo_zero`, `robust`, `sft`, combinable with weights) and `KTOTrainer`; `ORPOTrainer`, `CPOTrainer`, `OnlineDPOTrainer` are experimental. Defaults to know: `beta=0.1`; `learning_rate=1e-6` (full FT; about 1e-5 is suggested with adapters).

## 6.6 Where do the pairs come from?

| Source | How | Strengths | Weaknesses |
| --- | --- | --- | --- |
| **Verifier** | correct vs incorrect sample from the model | exact labels, on-policy | only for checkable tasks |
| **Human edits** | edited = chosen, original = rejected | high quality signal of what to change | cost |
| **Judge model** (RLAIF) | strong LLM ranks two answers with a rubric; **randomise order**, average both orders | scalable | judge biases (position, length, self-preference); needs calibration |
| **Model variants** | $\text{best-of-}N$ vs $\text{worst-of-}N$ under a reward | cheap | depends on reward quality |
| **Public datasets** | UltraFeedback etc. | ready | not your distribution; licenses |

**Good pair design:** same prompt; both responses *plausible* (hard negatives); difference attributable to one property; balanced lengths (or control for it); diverse prompts; include your safety/format rules. Trivial negatives teach little (lab 04 exercise 3).

## 6.7 Failure modes

| Failure | Mechanism | Symptom | Mitigation |
| --- | --- | --- | --- |
| **Likelihood displacement** | loss can be reduced by lowering both probabilities | `logps/chosen` drops; fluency degrades | add SFT term (`loss_type=["sigmoid","sft"]`), lower LR, larger $\beta$, IPO |
| **Length exploitation** | longer = larger log-ratio magnitude / judges prefer long | answers grow | length-normalised losses (SimPO-style), length-controlled eval, balanced pairs |
| **Off-policy mismatch** | pairs from another model | rewards saturate without behaviour change | on-policy or iterate sampling |
| **Reward over-optimisation** | proxy reward != truth | held-out true metric falls | smaller LR/steps, larger $\beta$, early stop by true metric |
| **Noisy labels** | annotator disagreement | accuracy plateau | robust loss (`label_smoothing`), filter by agreement |
| **Reference drift** | starting from a non-SFT'd model | unstable | do SFT first, then DPO |

## 6.8 Procedure

1. SFT first; freeze that checkpoint as reference.
2. Build 1-10k pairs (start small, e.g. 500) with a verifier or rubric; hold out 10%.
3. Run LoRA DPO with `beta=0.1`, LR `1e-5`, 1 epoch. Watch `rewards/accuracies`, `rewards/margins`, `logps/chosen`, response length.
4. Evaluate on the **task metric and general regression set**, not on reward margin.
5. If it helps: sample new pairs from the DPO model and repeat once or twice (iterative DPO). If it hurts: smaller LR/steps, larger $\beta$, SFT term.

## 6.9 Pros and cons

**Pros:** no rollout loop during training; stable; cheap; works with LoRA; directly optimises comparisons.
**Cons:** cannot explore beyond the pairs' support; sensitive to pair quality and length; implicit reward is uncalibrated outside training distribution; it can reduce likelihood of good answers; for *verifiable* tasks online RL or rejection sampling usually surpasses it.

## Checkpoint

* You can derive DPO from the KL-regularised objective and explain why $Z(x)$ cancels.
* You can build verifier-based pairs and read DPO training metrics.
* You can choose among DPO / IPO / KTO / ORPO / SimPO for a given data and memory situation.
