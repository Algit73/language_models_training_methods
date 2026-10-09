# 4. Supervised Fine-Tuning, LoRA and QLoRA

**Goal:** fine-tune a small model on your task reliably on 12 GB, understanding the maths well enough to choose rank, learning rate and target layers on purpose.
**Lab/script:** `labs/03_sft_and_lora_from_scratch`, `scripts/01_sft_lora_trl`, `scripts/04_generate_and_score`.

## 4.1 SFT in one equation

Given pairs $(x,y)$ and the concatenated token sequence $s=[x;y]$:
$$\mathcal{L}_{SFT}(\theta)=-\frac{1}{|M|}\sum_{t\in M}\log p_\theta(s_t\mid s_{<t})$$
$M$ = positions of the response (plus EOS). It is ordinary cross-entropy - *behaviour cloning*: it raises the probability of the demonstrated tokens and nothing else.

**Detail that changes results:** how the mean is taken. *Token-mean over the batch* weights long answers more; *per-sequence mean* weights every example equally. Frameworks differ (gradient accumulation with padded batches can also change the effective weighting unless the normaliser is the total token count). Recent TRL versions use the total non-masked tokens and compute the loss in chunks so the full vocab $\times$ length logits tensor is not materialised (`loss_type="chunked_nll"`, default). TRL also exposes a "Dynamic Fine-Tuning" (`dft`) loss that reweights tokens by their probability, motivated by the view of SFT as a poorly-shaped policy gradient - experimental territory, try it only as an ablation.

**What SFT can and cannot do**

| Can | Cannot (well) |
| --- | --- |
| teach output format, tone, tool syntax | add large amounts of new *knowledge* reliably (use RAG/CPT) |
| make a base model follow instructions | fix a missing capability that a verifier-driven RL step could improve |
| distill a teacher's behaviour | correct errors it has no signal about (it imitates mistakes present in data) |
| be stable and cheap | generalise beyond the demonstration distribution as well as RL often does (an active research debate) |

## 4.2 Full fine-tuning vs parameter-efficient fine-tuning (PEFT)

| Method | Trainable | Memory (per base param, weights+opt) | Pros | Cons |
| --- | --- | --- | --- | --- |
| Full FT | all | ~16 B | most capacity; simplest | impossible above ~0.5B on 12 GB; forgets more |
| LoRA | low-rank adapters | 2 B + 16 B per adapter param | tiny checkpoints, mergeable, less forgetting, many adapters per base | capacity limited by rank on large datasets |
| QLoRA | adapters on 4-bit base | ~0.56 B + adapters | fits 4-8B on 12 GB | slower step (dequantise), quantisation error in base, backend support |
| Prompt/prefix tuning | virtual tokens | tiny | very small | usually weaker for generation tasks |
| Adapters (bottleneck) | inserted layers | small | classic | adds inference latency unless fused |

## 4.3 LoRA mathematics

For a frozen weight $W_0\in\mathbb{R}^{d_{out}\times d_{in}}$, learn a low-rank update:
$$W=W_0+\frac{\alpha}{r}BA,\qquad A\in\mathbb{R}^{r\times d_{in}},\ B\in\mathbb{R}^{d_{out}\times r},\ r\ll\min(d_{in},d_{out})$$
Forward: $y=W_0x+\frac{\alpha}{r}B(Ax)$ - never form $BA$ during training.

* **Initialisation:** $A$ random (Kaiming-uniform), $B=0$, so the adapted model *equals* the base at step 0.
* **Parameters:** $r(d_{in}+d_{out})$ per matrix instead of $d_{in}d_{out}$. For a $2048\times6144$ MLP matrix and $r=16$: 131k vs 12.6M (1%).
* **Gradients.** Let $g=\partial\mathcal{L}/\partial y$ and $h=Ax$, $s=\alpha/r$:
  $$\frac{\partial\mathcal{L}}{\partial B}=s\,g\,h^\top,\qquad \frac{\partial\mathcal{L}}{\partial A}=s\,B^\top g\,x^\top$$
  At init $B=0\Rightarrow\partial\mathcal{L}/\partial A=0$: first only $B$ moves, then $A$ starts to matter as $B$ grows. This built-in warmup is why the best learning rate depends on run length.
* **Scaling $\alpha/r$:** keeps the update magnitude roughly rank-independent, so the same learning rate works across ranks (Thinking Machines, *LoRA Without Regret*, 2025). **rsLoRA** argues for $\alpha/\sqrt r$ to stabilise high ranks (`use_rslora=True` in PEFT).
* **Merging:** $W_{merged}=W_0+sBA$ - identical function, no inference overhead (lab 03 verifies numerically).
* **Compute:** per matrix LoRA needs $2N^2+6NR$ multiply-adds vs $3N^2$ for full fine-tuning: about two-thirds of the FLOPs.
* **Rank intuition:** the update $\Delta W$ has at most rank $r$. Fine-tuning updates for narrow tasks are empirically low-rank; broad/large datasets need higher rank. Adapter capacity is measured in bits: a rough upper bound on learnable information is ~2 bits/parameter; datasets carry about 1 bit per token of first-epoch loss. If dataset information exceeds adapter capacity, LoRA learns slower than full FT.

### Evidence-based defaults (Thinking Machines, Sept 2025; read the post)

1. **Apply LoRA to all linear layers**, especially the MLP. Attention-only LoRA underperformed MLP-only LoRA even with equal parameter counts.
2. **Learning rate about 10x the full-FT value** (fitted multiplier ~9.8; about 15x for very short runs). Optimal LR is nearly independent of rank with $\alpha/r$ scaling.
3. LoRA matches full fine-tuning for **small-to-medium SFT datasets**; it lags when data far exceeds capacity.
4. LoRA tolerates **large batch sizes worse** than full FT; prefer smaller batches (e.g. 16-64 effective) over huge accumulation.
5. For **RL**, even rank-1 LoRA matched full fine-tuning in their experiments, because policy-gradient RL absorbs only $O(1)$ bits per episode (chapter 7).

### Variants (know what they are; try only after the baseline works)

| Variant | Idea | When to try |
| --- | --- | --- |
| rsLoRA | scale by $\alpha/\sqrt r$ | ranks >= 64 |
| LoRA+ | larger LR for $B$ than $A$ | faster convergence (equivalent to changing init/LR timescale per TM's analysis) |
| DoRA | split weight into magnitude and direction; LoRA on the direction | small gains at some cost; check kernel support |
| PiSSA | initialise adapters from the top singular components of $W_0$ | faster early convergence |
| LoftQ | initialise to minimise quantisation error for QLoRA | when QLoRA quality lags 16-bit LoRA |
| Multi-adapter / merging | combine adapters (`add_weighted_adapter`) | multi-task from one base |

(Cited by title/authors/year in chapter 16; confirm details in the original papers before relying on them.)

## 4.4 QLoRA

Dettmers et al. (2023): quantise the frozen base to **4-bit NormalFloat (NF4)**, backprop through it into LoRA adapters kept in 16-bit.

* **NF4:** a 4-bit code whose 16 levels are quantiles of a standard normal, matching the roughly Gaussian distribution of pretrained weights. Weights are split into blocks (64 values) and normalised by a per-block scale (absmax).
* **Double quantisation:** quantise the per-block scales too (8-bit), lowering overhead from ~0.5 to ~0.127 bits per parameter.
* **Paged optimizers:** spill optimizer state to CPU RAM under memory pressure.
* Forward pass dequantises blocks on the fly to bf16 and multiplies: **slower than 16-bit LoRA**, but weights take ~0.5 B/param.
* The adapter gets gradients through the dequantised weights; the quantised weights are never updated.
* Practical: on XPU use the `bitsandbytes` XPU backend (documented as supported; check the exact version notes). If it is unavailable or unstable, use 8-bit or bf16 LoRA on smaller models. **Merge into the 16-bit base, not the quantised one.**

## 4.5 Hyperparameters and procedure (step by step)

1. **Baselines.** Evaluate prompting (few-shot) and the untouched model on your frozen dev set with greedy decoding.
2. **Smoke test.** 20 examples, 1 epoch, tiny `max_length`: confirm the device is XPU, masks are right, loss decreases.
3. **First real run.** LoRA `r=16, alpha=32, dropout=0`, all-linear; LR `2e-4` (LoRA) - or `2e-5` if you run full FT; cosine/linear decay with ~3-5% warmup; effective batch 16-32 via accumulation; 1-3 epochs; `max_length` just above your 95th-percentile example; bf16; gradient checkpointing on.
4. **Watch:** training loss (smooth decline), grad-norm spikes, dev loss *and* dev task metric (the metric matters, loss can keep falling while behaviour degrades).
5. **Sweep one axis at a time:** LR $\times\{0.5,1,2\}$, then rank $\{8,16,64\}$, then epochs, then data size.
6. **Inspect** 20 failures; fix data before hyperparameters.
7. **Regression-test** on general prompts to catch forgetting.
8. **Freeze** the adapter and record config, versions, seeds, data hashes.

| Hyperparameter | Start | Symptom if too high | Symptom if too low |
| --- | --- | --- | --- |
| LR (LoRA) | 1e-4 to 3e-4 | loss spikes, gibberish, forgetting | slow loss decline, no behaviour change |
| Rank $r$ | 16 | - (only cost) | underfits large datasets |
| Epochs | 1-3 | memorisation, dev metric falls | underfit |
| Effective batch | 16-32 | LoRA plateaus (batch penalty) | noisy training |
| `max_length` | p95 of data | OOM | truncated answers |
| Dropout | 0-0.05 | underfit | overfit small data |

## 4.6 Common failures (symptom -> cause -> fix)

| Symptom | Cause | Fix |
| --- | --- | --- |
| Train loss low, dev metric poor | memorisation or leakage | dedupe, group split, fewer epochs, more diverse data |
| Output never stops / rambles | EOS missing in target or wrong template | add EOS/turn-end token; check `eos_token` |
| Output repeats the prompt | no masking | assistant/completion-only loss |
| Format perfect but wrong content | SFT learned style, not skill | add verifier-filtered data, then RL/rejection sampling |
| General chat ability worsens | narrow data + high LR | LoRA, lower LR, 5-20% replay data |
| Good in notebook, bad in app | template mismatch or different decoding | use the identical chat template and decoding at inference |
| XPU OOM with only 1B model | logits/activation, long sequence | shorten, checkpoint, chunked loss |

## 4.7 Pros and cons summary

**Use SFT+LoRA when** you know the desired outputs, need consistent format/behaviour, want cheap reproducible iteration, and want to ship a small adapter.
**Do not rely on it when** the skill is better verified than demonstrated (math/code/agents -> add chapters 7-8), or when the need is fresh facts (RAG).

## Checkpoint

* A LoRA adapter that beats prompt-only and the base on the frozen dev set, with a 95% CI (lab 08).
* You can explain the gradient of $A$ and $B$, the $\alpha/r$ scale, merging, and why all-linear + ~10x LR are the defaults.
* You can estimate whether a 4B QLoRA run at your `max_length` fits (lab 01).
