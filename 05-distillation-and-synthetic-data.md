# 5. Distillation and Synthetic Data

**Goal:** use a stronger model (or a verifier) as a teacher to produce a stronger small model, and know which flavour of distillation to choose.
**Lab:** `labs/07_distillation_forward_vs_reverse_kl`.

## 5.1 Why distillation matters for small models

A small model cannot discover everything a large one knows, but it can *imitate* a large model's behaviour far more cheaply than it can learn it alone. Reported pattern (e.g. in the DeepSeek-R1 paper, arXiv 2501.12948): **distilling reasoning traces from a large RL-trained model into a small model outperformed running RL directly on the small model** at comparable cost. The general principle: *teacher signal is dense and cheap; RL signal is sparse and expensive.*

## 5.2 The mathematics

Teacher $p(y_t\mid y_{<t},x)$, student $q_\phi$. Per-token divergences:

$$\text{Forward KL: }D_{KL}(p\Vert q)=\sum_y p(y)\log\frac{p(y)}{q(y)}\qquad \text{Reverse KL: }D_{KL}(q\Vert p)=\sum_y q(y)\log\frac{q(y)}{p(y)}$$

* **Forward KL (mass-covering).** $q$ must cover all modes of $p$. Minimising it on teacher *samples* is exactly maximum likelihood = cross-entropy training. A too-small student spreads mass between modes (lab 07 shows mass in the valley) - at generation time it blends styles.
* **Reverse KL (mode-seeking).** $q$ is penalised for mass where $p$ has none, so it commits to high-probability modes: sharper samples, less diversity. It requires samples *from the student*, evaluated by the teacher - hence **on-policy**.
* **Temperature** softens teacher distributions: $p_\tau=\mathrm{softmax}(z/\tau)$. The soft non-argmax probabilities carry "dark knowledge" (Hinton et al., 2015). Multiply by $\tau^2$ to keep gradient scale comparable.
* **Jensen-Shannon family:** $\mathrm{JSD}_\beta=\beta D_{KL}(p\Vert m)+(1-\beta)D_{KL}(q\Vert m)$, $m=\beta p+(1-\beta)q$, interpolates the two (GKD, Agarwal et al. 2023).
* **Mixed loss** used in practice: $\mathcal{L}=(1-\lambda)\mathcal{L}_{CE}(y^*)+\lambda\tau^2D_{KL}(p_\tau\Vert q_\tau)$.

## 5.3 Three practical families

| Family | Data | Loss | Needs teacher weights? | Pros | Cons |
| --- | --- | --- | --- | --- | --- |
| **Sequence-level / "SFT on teacher outputs"** | teacher-generated text | cross-entropy | no (API/local sampling) | simplest; works across tokenizers | exposure bias; loses teacher's uncertainty; licensing |
| **Logit (white-box) KD** | fixed text (teacher or real) | forward KL on logits | yes; same tokenizer/vocab | richer signal per token | memory for both models; vocab match needed |
| **On-policy distillation (GKD, MiniLLM, related)** | *student-generated* text scored by teacher | reverse KL / JSD (+ policy gradient in MiniLLM) | yes | fixes train/test mismatch; teaches recovery from own mistakes; dense reward | needs sampling loop; teacher forward passes cost memory |

TRL currently lists a stable `DistillationTrainer` and experimental `GKDTrainer`, `MiniLLMTrainer`, `GOLDTrainer` and others; their availability and arguments change quickly, so **read the current docs** and check whether your teacher/student tokenizers must match. There is also a Thinking Machines post on on-policy distillation in their *Connectionism* series.

### Why on-policy helps (intuition)
SFT on teacher text shows the student only *states the teacher visits*. At inference the student drifts into states it never saw; errors compound (chapter 10's $p^T$ argument). On-policy distillation trains on the student's own trajectories with the teacher's judgement at each token: it is like RL with a **dense, per-token reward** $\log p(y_t)-\log q(y_t)$ instead of one outcome bit - far more sample-efficient than GRPO when you have a teacher.

## 5.4 Reasoning distillation recipe (small model)

1. Choose an **open-weights teacher with a permissive license** that supports the use you intend. Read its license for any restriction on using outputs to train other models.
2. Collect prompts in your domain (not just public benchmarks).
3. Sample $k\ge2$ traces per prompt at moderate temperature.
4. **Filter with verifiers**: final answer correct, format valid, length <= budget, no contamination of your dev set. Optionally keep the *shortest* correct trace.
5. Deduplicate; balance by difficulty/topic.
6. SFT the student (LoRA at first) on `prompt -> reasoning + answer`.
7. Evaluate pass@1 at fixed length; compare to base, to prompting, and to RL from the same start.
8. Optionally follow with RLVR (chapter 7) for robustness.

## 5.5 Synthetic-data quality controls

* **Verifier first, teacher second.** If a checker exists, teacher correctness is irrelevant: keep only verified samples.
* **Diversity:** cluster prompts/answers; cap per cluster; vary templates and personas; avoid mode collapse.
* **Adversarial/hard cases:** mine the student's failures, ask the teacher for targeted examples (error-driven data loop).
* **Audit:** read 100 random samples per dataset version.
* **Provenance:** record teacher, prompt, temperature, date, and license per example.

## 5.6 Risks

| Risk | Mitigation |
| --- | --- |
| Teacher errors inherited | verifier filter; multiple teachers; cross-check |
| Teacher style overfit (verbosity) | length constraints; style-normalise; evaluate length |
| Tokenizer mismatch for logit KD | use sequence-level KD, or a method designed for cross-tokenizer alignment (check docs) |
| License/terms violations | check each teacher's terms before using outputs for training |
| Eval contamination via teacher memory | decontaminate; build private evals |
| Model collapse by recursive self-training | keep real data share, verifier filters, fresh prompts each round |

## Checkpoint

* You can explain forward vs reverse KL behaviour with the lab 07 numbers.
* You can choose between sequence-level KD, logit KD and on-policy distillation for a given teacher/student pair.
* You have a verifier-filtered distillation pipeline plan with provenance fields.
