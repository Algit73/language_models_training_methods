# 3. Data, Tokenization and Chat Templates

**Goal:** turn raw material into training data that teaches exactly the behaviour you want, with clean splits and correct masking. Most failed fine-tunes are data failures.

## 3.1 What "data" means at each stage

| Stage | Data | Size (typical for small models) | What it teaches |
| --- | --- | --- | --- |
| Continued pretraining (CPT) | raw domain text | 10^8 - 10^10 tokens | vocabulary, facts, style of a domain |
| SFT | (prompt, ideal response) | 10^2 - 10^5 examples | format, task behaviour, tool use |
| Preference | (prompt, chosen, rejected) | 10^3 - 10^5 pairs | relative quality, style, safety |
| RL / verifiers | (prompt, checker) | 10^3 - 10^5 prompts | *robust success* rather than imitation |
| Evaluation | frozen held-out set with checker | 10^2 - 10^3 | truth about progress |

**Quality beats quantity** for post-training. The LIMA paper (Zhou et al., 2023) showed strong instruction-following from about one thousand carefully curated examples, arguing that alignment mostly teaches *format and style* on top of knowledge learned in pretraining. The LoRA-capacity view agrees: post-training datasets carry few bits compared with adapter capacity (chapter 4).

## 3.2 The task contract (write this first)

A one-page document that decides everything else:

1. **Input**: what text/images/tools the model sees; maximum lengths.
2. **Output**: exact format (JSON schema, code patch, action grammar, tone/length).
3. **Success**: the metric and its threshold; how a checker decides.
4. **Failure modes**: what must *never* happen (PII leak, destructive action, invented citation).
5. **Abstention**: when the model should say "I don't know" or ask a question.
6. **Out of scope**.

## 3.3 Formats (JSONL)

One JSON object per line; give every example an `id` and a `source`.

```json
// SFT - conversational prompt/completion (TRL computes loss on completion by default)
{"id":"ex-001","prompt":[{"role":"user","content":"Extract the deadline: Apply by 12 May 2027."}],
 "completion":[{"role":"assistant","content":"{\"deadline\":\"2027-05-12\"}"}]}

// Preference pair (explicit prompt)
{"id":"pr-001","prompt":[{"role":"user","content":"..."}],
 "chosen":[{"role":"assistant","content":"..."}],"rejected":[{"role":"assistant","content":"..."}]}

// Verifiable RL prompt: extra columns are passed to reward functions
{"id":"rl-001","prompt":[{"role":"user","content":"Compute 17*23. Give \\boxed{answer}."}],"ground_truth":"391"}

// Tool-calling SFT (messages include tool calls and tool results; tools column = JSON schemas)
{"id":"tc-001","messages":[...],"tools":[{"type":"function","function":{"name":"sql","parameters":{}}}]}

// Agent trajectory (one step = one training example, or the whole episode as a multi-turn chat)
{"id":"ag-001","task":"enable dark mode","step":0,"observation":"<a11y tree>","action":{"action":"click","target":"sw_dark"},"result":"ok"}
```

## 3.4 Chat templates, special tokens and masking - where silent bugs live

A **chat template** (Jinja, stored in the tokenizer) converts messages to the exact token string the model was trained with. Rules:

* Always build training text with `tokenizer.apply_chat_template` (or let TRL do it), never by hand.
* The **generation prompt** (`add_generation_prompt=True`) and the training prefix must match what you will use at inference.
* The **EOS/turn-end token** must be in the training target so the model learns to stop. (For some base models you must set `eos_token` in `SFTConfig` to the chat end token, e.g. `<|im_end|>`.)
* **Loss masking:** train only on what the model must produce. In SFT that is the assistant tokens:
  $$
  \mathcal{L}=-\frac{1}{|M|}\sum_{t\in M}\log p_\theta(x_t\mid x_{<t}),\quad M=\text{assistant positions}
  $$
  TRL offers `completion_only_loss` (prompt-completion data) and `assistant_only_loss=True` (chat data; needs `{% generation %}` markers in the template, which TRL patches for some families). Without masking you spend capacity modelling the prompt and, worse, teach the model to imitate the user.
* **Thinking models:** some templates add or strip reasoning blocks (`<think>...`). Verify how the template treats them in training vs inference.
* Verify by decoding: take 5 training examples, print the token ids with the masked positions marked. Do this before every run.

## 3.5 Packing, padding and length

* **Padding** wastes compute. **Packing** concatenates several examples into one fixed-length sequence (with attention boundaries or position resets when supported), usually 1.5-3x faster for short examples. TRL: `packing=True` (strategies `bfd`, `bfd_split`, `wrapped`).
* Sort/bucket by length if you cannot pack.
* **Truncation** silently cuts the end - which is where your answer/EOS lives. Filter over-length examples instead of truncating them. For VLMs truncation can remove image tokens (use `max_length=None`).
* Token budget: $D_{\text{tokens}}=\sum_i |x_i|\times\text{epochs}$. Estimate training time with $6ND$ (lab 01).

## 3.6 Cleaning pipeline

```mermaid
flowchart LR
  R[Raw] --> L[License + PII filter]
  L --> N[Normalise: unicode, whitespace, HTML]
  N --> Q[Quality filters: length, language, perplexity, rules]
  Q --> D[Deduplicate: exact + near-duplicate MinHash]
  D --> X[Decontaminate vs eval sets]
  X --> S[Split by group]
  S --> V[Validate: template, masks, schema]
```

* **Near-duplicate removal** (MinHash-LSH over word n-gram shingles) removes repeated boilerplate and prevents memorisation. Exact-hash dedup alone misses most.
* **Decontamination:** remove any training document sharing long n-grams (e.g. 8-13 words) with any evaluation item (lab 08 has a minimal checker).
* **PII and secrets:** scan for emails, keys, phone numbers, tokens; redact or drop. Never put secrets in a dataset you might publish.
* **Licensing:** record licenses per source. Distilling from a proprietary model's outputs may violate its terms; check before using such outputs for training.

## 3.7 Splitting without leakage

Random row splits leak. Split by the unit that shares information:

| Data | Split by |
| --- | --- |
| support tickets, emails | customer / thread |
| code | repository (and time) |
| documents | document ID / author |
| GUI tasks | task *template* and app state, not screenshot |
| synthetic variants of one seed | the seed |

Keep **train / dev / test**: dev for decisions, test touched once per milestone. Freeze them by hash of the group ID so they never change.

## 3.8 Synthetic data

A strong model or a program can generate data when humans are too slow:

* **Self-Instruct** (Wang et al., 2022): bootstrap instructions from a few seeds with a model; filter; repeat.
* **Distillation:** a stronger *teacher* writes answers/rationales (chapter 5).
* **Programmatic generation** with a verifier (arithmetic, SQL, unit tests, GUI sandboxes): labels are exact (labs 05, 09, 10).
* **Rejection sampling:** sample many, keep verified (lab 05).
* **Back-translation / paraphrase** for robustness.

Pitfalls: *model collapse* from training repeatedly on your own unfiltered outputs, loss of diversity, inherited teacher errors and biases, and leakage of the teacher's evaluation contamination. Mitigate with verifiers, diversity constraints (dedupe, stratify by topic/difficulty), and a **human-audited sample** of at least a few hundred items.

## 3.9 How much data, and preventing forgetting

* Start tiny: 50-200 excellent examples + 20 hard ones, train, read failures, add targeted data. Scale only when the curve says more data helps.
* LoRA capacity (chapter 4): about 1 bit per token of first-epoch loss bounds the information you can memorise; small behaviour changes are far below that, memorising a knowledge base is not - use RAG for facts (chapter 13).
* **Catastrophic forgetting:** narrow fine-tuning can erode general ability. Mitigate with low learning rate, LoRA (frozen base), 5-20% *replay* of general instruction data, early stopping, and regression tests on general benchmarks.
* **Epochs:** 1-3 for SFT. Repeating small data many times memorises; watch train vs dev gap.

## 3.10 Pros and cons of common data sources

| Source | Pros | Cons |
| --- | --- | --- |
| Human-written | highest quality, real distribution | slow/expensive; inconsistent without a guide |
| Logs of real use | realistic | privacy, noisy, biased to current behaviour |
| Teacher-LLM generated | scalable, fluent | inherits errors; licensing; low diversity |
| Programmatic + verifier | exact labels, infinite | may not match the real distribution |
| Public datasets | free, benchmarked | contamination, licenses, distribution mismatch |

## 3.11 Data card (keep one per dataset)

Sources and licenses; collection dates; filtering steps and counts after each step; split method and sizes; known gaps/biases; PII handling; annotation guide version; owner and contact; hashes of files.

## Checkpoint

* Your contract, schema, split rule and masking are written down and you have decoded 5 examples with masks visible.
* Your eval set is frozen and decontaminated.
* You can explain when to use RAG instead of fine-tuning.
