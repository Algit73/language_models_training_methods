# 2. Data and Traditional Post-Training

## Goal

Turn a base model into a narrow specialist using disciplined data work and supervised fine-tuning (SFT).

## The methods, in order

1. **Prompting and retrieval**: change no weights. Use this as the baseline.
2. **Continued pretraining (CPT)**: train next-token prediction on unlabeled domain text. Use it for vocabulary/style/domain adaptation, not instruction following.
3. **SFT**: train input-to-desired-answer pairs. This is the default method for behavior, format, and task competence.
4. **Parameter-efficient fine-tuning (PEFT)**: train adapters, especially LoRA, instead of all weights.
5. **QLoRA**: keep a quantized base model while training LoRA adapters. This is typically the practical choice on 12 GB.

## Lesson: what SFT and LoRA optimize

Run `python3 labs/02_sft_data_and_loss.py` first. It computes next-token cross-entropy on a tiny vocabulary and shows why one correct target must be associated with each prompt.

For a token sequence $x_1, \ldots, x_T$, causal-language-model SFT minimizes negative log likelihood over target tokens:

$$\mathcal{L}_{SFT} = -\frac{1}{|M|}\sum_{t \in M}\log p_\theta(x_t \mid x_{<t})$$

$M$ is the set of assistant-response positions. Response masking excludes system and user prompt tokens from the loss. Without it, the model also learns to predict the prompt, spending capacity on behavior you do not need.

For an existing weight matrix $W_0 \in \mathbb{R}^{d_{out} \times d_{in}}$, LoRA freezes $W_0$ and learns:

$$W = W_0 + \frac{\alpha}{r}BA$$

where $A \in \mathbb{R}^{r \times d_{in}}$, $B \in \mathbb{R}^{d_{out} \times r}$, and $r$ is much smaller than either hidden dimension. Instead of $d_{out}d_{in}$ trainable parameters, LoRA has $r(d_{in}+d_{out})$.

### Step by step: first real SFT run

1. Select one narrow task with a checkable output, such as extracting a deadline into JSON.
2. Write 100 examples by following one annotation guide. Include 20 deliberately difficult cases.
3. Reserve 20% by source document, not random row, for validation/test.
4. Evaluate the frozen model with the deployment prompt and save every output.
5. Format train records with the model's own chat template. Mask loss to assistant tokens.
6. Train rank 8 LoRA at sequence length 512, batch size 1, gradient accumulation 8, for one epoch.
7. Compare base and adapter on the frozen test set using schema validity and field F1.
8. Inspect every failure before increasing epochs, rank, or dataset size.

### When to choose each method

| Method | Pros | Cons | Use it when |
| --- | --- | --- | --- |
| Prompting/RAG | no training, fast updates | can be inconsistent | knowledge is external or task is already solved |
| CPT | adapts vocabulary/domain distributions | needs lots of licensed text; can harm instruction behavior | domain language is the main gap |
| Full fine-tuning | maximum flexibility | optimizer/gradient memory is impractical here | only tiny models or rented compute |
| LoRA | cheap, portable adapters, strong baseline | rank/target modules still need tuning | default on the B580 |
| QLoRA | reduces base-weight memory further | backend quantization support may vary on XPU | LoRA model does not otherwise fit |

## Build the dataset before the trainer

Use JSONL with one example per line. Keep a stable ID and split indicator.

```json
{"id":"support-001","messages":[{"role":"system","content":"Return only valid JSON."},{"role":"user","content":"Extract the deadline from: Apply by 12 May 2027."},{"role":"assistant","content":"{\"deadline\":\"2027-05-12\"}"}],"source":"curated","split":"train"}
```

### Data workflow

1. Write a task contract: user input, permitted context, desired output, safety limits, and metric.
2. Collect examples with permission. Remove secrets, personal data, duplicates, boilerplate, and leakage from evaluation examples.
3. Define an annotation guide with positive and negative examples.
4. Create train/validation/test splits by document, user, repository, or time boundary. Random row splits often leak near duplicates.
5. Write 50-200 high-quality examples before scaling. For narrow tasks, quality beats volume.
6. Add hard cases from baseline failures.

## SFT recipe

1. Evaluate the untouched base model with the exact chat template you will use later.
2. Train LoRA/QLoRA on the assistant response tokens only when your framework supports response masking.
3. Start with 1-3 epochs; watch validation loss and task metric. More epochs are not automatically better.
4. Save the adapter and a model card containing base revision, data license, intended use, limitations, and evaluation.
5. Compare base, prompt-only, and tuned model on the held-out test set.

## Common failures

| Symptom | Likely cause | First repair |
| --- | --- | --- |
| Great train loss, poor test quality | memorization or leakage | deduplicate; improve split; reduce epochs |
| Model ignores output format | inconsistent targets/template | normalize targets and train with deployment template |
| General ability worsens | narrow/low-quality data or high learning rate | lower LR; mix small general data; use smaller adapter |
| Responses become repetitive | long training or decoding issue | inspect data; stop earlier; tune generation settings |

## Milestone

Ship a LoRA adapter that beats the frozen base model on a held-out, task-specific metric by a predeclared margin. Keep the failure examples; they become the next dataset revision.
