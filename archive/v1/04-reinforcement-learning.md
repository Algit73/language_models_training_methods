# 4. Reinforcement Learning Methods

## Goal

Use RL only when supervised targets are incomplete but you can express preferences or success as rewards. First make SFT work: RL from a weak, random policy wastes compute and amplifies reward bugs.

## Core vocabulary

- **Policy** $\pi_\theta$: the model being optimized.
- **Reward** $r$: a scalar signal for an output or trajectory.
- **Reference policy**: usually the SFT model; constrains drift.
- **KL penalty**: discourages the policy from moving too far from the reference.
- **Reward hacking**: achieving a high reward without accomplishing the intended task.

## Lesson: the algorithms and their mathematics

Run `python3 labs/04_preference_and_policy_optimization.py`. It has two safe toy calculations: a DPO pairwise loss and normalized group rewards. It does not update a language model; it helps you predict the sign and scale of the signals a trainer consumes.

### DPO

For prompt $x$, preferred answer $y_w$, rejected answer $y_l$, reference policy $\pi_{ref}$, and temperature-like coefficient $\beta$, DPO minimizes:

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta\left[\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right]\right)$$

It raises the chosen answer's relative likelihood without separately fitting a reward model. It is usually the first preference method to try on one GPU.

### PPO

PPO starts from sampled rollouts and an advantage estimate $A_t$. With probability ratio $r_t(\theta)=\pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$, its clipped objective is:

$$\mathcal{L}_{PPO}=\mathbb{E}_t[\min(r_tA_t,\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]$$

The clip limits an update that changes the policy too abruptly. Practical LLM PPO also needs a value model, KL control, rollouts, and careful logging; that is why it is not your starting method.

### GRPO-style group methods

For several completions of the same prompt with rewards $r_1,\ldots,r_G$, a simple group-relative signal is:

$$A_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)+\varepsilon}$$

The method compares siblings from the same prompt rather than relying on a learned critic. This can reduce memory relative to PPO variants, but still requires generating groups and trustworthy rewards.

### Step by step: selecting and running a method

1. If you have a verifier, begin with rejection sampling.
2. If you have reviewed chosen/rejected pairs, use DPO from an SFT checkpoint.
3. If you have an online sandbox and a tested scalar reward, try small group-based RL.
4. Use PPO only after you can explain your reward, KL term, rollout budget, and value-model memory.
5. Before every run, score a frozen test set, unit-test the reward, and log reward components separately.
6. During every run, inspect high-reward outputs for shortcuts, not only the reward curve.

### Pros and cons

| Method | Pros | Cons |
| --- | --- | --- |
| DPO | offline, stable, no reward model or rollout phase | cannot discover beyond the quality/diversity of preference pairs |
| PPO | flexible for arbitrary sequential rewards | high memory/complexity; unstable reward optimization |
| GRPO-style | no separate critic in common variants; good for verifiable outputs | expensive sampling; reward hacking remains possible |

## Choose the lightest method

| Method | Feedback required | Why use it | B580 suitability |
| --- | --- | --- | --- |
| Rejection sampling | deterministic verifier | simplest verified improvement | excellent |
| DPO / ORPO / SimPO | chosen vs rejected pairs | preference alignment without online rollout | excellent |
| KTO | binary desirable/undesirable labels | sparse feedback | good |
| PPO | scalar reward plus rollouts/value model | general but operationally heavier | small experiments only |
| GRPO-style group methods | multiple sampled answers and scalar reward | removes/avoids a separate critic in variants | good for short verifiable tasks |

Exact implementations and names change quickly. Read your trainer’s algorithm documentation and paper, then validate behavior with a toy reward before a real run.

## Preference optimization: best first RL-like project

Create pairs with the same prompt:

```json
{"prompt":"Return a JSON object with title and date.","chosen":"{\"title\":\"Meeting\",\"date\":\"2027-05-12\"}","rejected":"The meeting is May 12.","reason":"chosen is schema-valid"}
```

1. Collect pairs from human edits, a deterministic validator, or carefully reviewed model candidates.
2. Keep prompts diverse; ensure the rejected answer is plausible, not absurdly bad.
3. Run DPO (or the repository’s supported preference method) from your SFT adapter.
4. Measure held-out preference win rate *and* task metrics; a reward improvement alone is insufficient.

## Online RL recipe

1. Write and unit-test the reward function. Log every reward component.
2. Start from a model with nonzero success rate.
3. Use short output limits, small batches, and a low learning rate.
4. Include a format reward, task-success reward, safety penalty, and KL constraint.
5. Evaluate at regular checkpoints with a frozen test set and inspect samples manually.
6. Stop when task success plateaus or quality outside the reward declines.

An illustrative shaped reward is:

$$r = 1.0r_{task} + 0.1r_{format} - 1.0r_{unsafe} - \beta\operatorname{KL}(\pi_\theta\Vert\pi_{ref})$$

The coefficients are hypotheses to test, not universal constants.

## Reward-safety checklist

- Sandbox executors and tools; never reward destructive actions.
- Separate hidden evaluation from reward development.
- Test adversarial shortcuts that satisfy surface metrics.
- Log prompts, actions, reward components, and terminal states.
- Keep human approval for consequential external actions.

## Milestone

On a verifier-backed task, run one preference-optimization experiment and one small online-RL or group-sampling experiment. Retain the method only if it beats SFT on held-out success without increasing invalid or unsafe outputs.
