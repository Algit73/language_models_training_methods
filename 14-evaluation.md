# 14. Evaluation

**Goal:** know *for certain* whether a change helped. Evaluation is the most valuable skill in this course: without it, training is guesswork.
**Lab/script:** `labs/08_evaluation_statistics`, `scripts/04_generate_and_score`.

## 14.1 Principles

1. **Define success before training**: metric, threshold, decoding settings.
2. **Freeze the test set first**, decontaminate it, touch it rarely.
3. **Baselines always**: base model, prompting, rules/simple models, previous version.
4. **Same conditions**: identical prompts, template, decoding, max tokens, number of samples.
5. **Statistics**: confidence intervals and paired tests; do not declare winners within noise.
6. **Read failures**: at least 20 raw outputs per run, categorised.
7. **Evaluate behaviour at the level users experience it** (closed loop for agents; sampled outputs, not just loss).

## 14.2 Metric toolbox

| Output type | Primary metric | Also |
| --- | --- | --- |
| Exact/structured | exact match, field F1, schema validity | error taxonomy |
| Numeric/reasoning | accuracy, pass@1 over $n$ samples | $\text{pass@}k$, mean tokens, truncation rate |
| Code | unit-test $\text{pass@}k$ | apply rate, security lint, hidden tests |
| Retrieval | Recall@k, MRR, nDCG | latency |
| Classification | macro-F1, per-class recall | calibration (ECE), confusion matrix |
| Free text | rubric score, pairwise win-rate, faithfulness | length, toxicity, PII leaks |
| Agents | closed-loop success | steps, invalid calls, violations, recovery |
| Speech | WER | per-speaker/accent breakdown |
| Language modelling | held-out loss / perplexity / bits-per-byte | only for same tokenizer/data |
| System | latency (time-to-first-token, tokens/s), memory, cost per 1k requests | p95/p99 |

## 14.3 Statistics you must know (lab 08)

* **Accuracy CI:** for $k$ successes in $n$ trials use the **Wilson interval** (better than $\pm1.96\sqrt{\hat p(1-\hat p)/n}$ for small $n$ or extreme $\hat p$). With $n=200$ and $\hat p=0.6$ the 95% CI is about $\pm0.07$.
* **Sample size to detect an improvement $\delta$** with independent samples (95% confidence, 80% power): $n\approx2\frac{(1.96+0.84)^2\bar p(1-\bar p)}{\delta^2}$ per model: about 2,400 items per model to detect +5 points at $p\approx0.6$.
* **Paired comparison:** run both models on the *same* items. Use the **paired bootstrap** CI for $\Delta$ accuracy, or **McNemar** on discordant pairs ($n_{10}$, $n_{01}$; under H0 $n_{01}\sim\mathrm{Bin}(n_{01}+n_{10},\tfrac12)$). Correlation between models makes paired tests much more powerful.
* **Sampling variance:** for stochastic decoding, average over multiple samples per item and over seeds. Two seeds are not enough to claim small differences; for RL runs seed-to-seed variance is large, so compare across several seeds when claims matter.
* **Multiple comparisons:** if you test 20 variants, one "wins" by chance. Decide on the primary comparison beforehand, or correct (Bonferroni: $\alpha/m$).
* **Overfitting to dev set:** repeated peeking leaks information. Use a dev set for choices and a **test set once** per milestone; refresh with new data periodically.

## 14.4 Contamination and leakage

* **Train-test overlap** inflates scores. Decontaminate by n-gram overlap (word 8-13-grams) and by near-duplicate search; check both the question and the answer.
* **Benchmark leakage via teachers:** a teacher model may have seen public test sets.
* **Temporal splits** (problems created after a model's data cutoff), **private sets**, and **programmatic generators** (new problems each run) reduce risk.
* **Template leakage:** near-identical items across splits (same template, different numbers) make generalisation look better than it is. Split by template.

## 14.5 Benchmarks and harnesses

* EleutherAI's **lm-evaluation-harness** runs many standard multiple-choice and generation tasks consistently; results depend on prompt format, few-shot count and scoring method, so **reproduce baselines before comparing to published numbers**.
* Domain benchmarks: see chapters 8-11 and 13. Treat public numbers as *weak evidence* for your use case.
* **Regression suite:** 50-200 general prompts/tasks with checkers (format, safety, basic knowledge) run after every fine-tune to catch forgetting.
* **Unit-test-style evals ("evals as tests"):** each behaviour requirement becomes a test with expected outcome and severity; gate releases on them.

## 14.6 LLM judges and human evaluation

See chapter 12.5 for biases and calibration. Human evaluation: written guidelines, 2-3 raters, measure agreement (kappa), adjudicate disagreements, sample representatively, blind the system identity. Use humans to *calibrate* automated metrics, then use the metrics for iteration.

## 14.7 Safety and robustness evals

* **Red-team prompts** for your domain: prompt injection, jailbreaks, PII extraction, destructive actions (agents), insecure code.
* **Robustness:** typos, paraphrase, different formatting, long/short inputs, language mixes, UI perturbations.
* **Calibration and abstention:** does the model say "not found/unknown" when appropriate?
* **Bias/fairness checks** relevant to your users.
* Report *violation rates with CIs*, not anecdotes.

## 14.8 Evaluation during training

* Log loss *and* the task metric every N steps on a small fixed dev subset (generation-based metrics are slower: use 100-200 items).
* **Checkpoint selection** by dev metric, not training loss; keep the best and the last.
* For RL: track **held-out** reward separately from training reward (they diverge when hacking or overfitting).
* Early stop when the dev metric plateaus for a preset number of evaluations.

## 14.9 Experiment tracking and reproducibility

Record for each run: code commit, `pip freeze`, Windows driver + WSL kernel + Ubuntu version, torch/trl/peft versions, base model *revision hash*, dataset file hashes, random seeds, full config, decoding settings, hardware, peak memory, wall time, metrics with CIs, and links to sample outputs. Even a Markdown table or CSV works; MLflow/W&B are optional.

**Experiment log template**

```
run_id: 2026-10-03-sft-extract-r16
goal/hypothesis: rank 16 beats rank 8 on field-F1 by >= 2 points
base model@revision: Qwen/Qwen3-0.6B @ <hash>     adapter: r=16 alpha=32 all-linear
data: train-v3.jsonl (sha256 ...) n=3,200; dev-v3 n=400; test-v3 n=600 (frozen)
train: lr 2e-4, eff. batch 16, epochs 2, max_length 512, bf16, seed 0
hardware: Arc B580 12GB, WSL2 Ubuntu 24.04, driver <ver>, torch <ver>
cost: peak mem 7.9 GiB, 42 min
results: dev field-F1 0.914 [0.89,0.93]; base 0.802 [0.77,0.83]; paired delta +0.112 [0.08,0.14], McNemar p<1e-6
failures (20 inspected): 9 wrong date format, 6 missing second entity, 3 hallucinated city, 2 invalid JSON
decision: keep; next: add date-format hard cases
```

## 14.10 Common evaluation mistakes

| Mistake | Fix |
| --- | --- |
| Reporting training loss as success | report the task metric on held-out data |
| One sample, one seed | multiple samples/seeds, CIs |
| Changing prompt between models | fix the prompt |
| Tuning on the test set | dev/test discipline |
| Ignoring truncation | report truncation rate; same max tokens |
| Judge = trained model | independent judge, calibrated |
| Benchmarks only | add private task-specific eval |
| No regression tests | run the general suite every time |

## Checkpoint

* You can produce a 95% CI, a paired comparison and a contamination check for a two-model comparison.
* Your experiment log is complete enough for someone else (or you in six months) to reproduce a run.
