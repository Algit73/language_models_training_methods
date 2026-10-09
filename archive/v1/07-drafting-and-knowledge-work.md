# 7. Drafting and Knowledge-Work Models

## Goal

Build an assistant for controlled writing: summaries, emails, reports, meeting notes, edits, extraction, and transformation into structured formats.

## Why this is different

Writing quality is subjective, so use a rubric plus deterministic checks. A model that has the right tone but invents a source, leaks private data, or ignores requested constraints has failed.

## Lesson: separate style from groundedness

Run `python3 labs/07_grounded_extraction.py`. It evaluates whether each extracted action item is supported by a supplied note. This is deliberately strict: matching a good writing style cannot compensate for unsupported claims.

For a draft $y$ sourced from document $d$, use a decomposed score rather than a single vague rating:

$$R(y,d)=w_gG(y,d)+w_cC(y)+w_sS(y)-w_hH(y,d)$$

Here $G$ is groundedness, $C$ required-content coverage, $S$ style/rubric score, and $H$ unsupported or privacy-harming claims. The values should be measured independently; choose weights only after you decide which failure is unacceptable.

### Step by step: a grounded drafting model

1. Choose one output contract: input evidence, audience, fields, length, tone, permitted claims, and when to ask a question.
2. Gather permitted source-to-output pairs. Redact private data before it enters a dataset.
3. Annotate each target claim with its evidence span or source ID.
4. Create a held-out set split by author/client/document family.
5. SFT for the output contract. Evaluate field coverage, support, and schema validity separately.
6. Create preference pairs only where the reason is clear: supported vs invented, concise vs repetitive, or correctly redacted vs leaked.
7. Use DPO for those pairs only if its held-out groundedness gain exceeds the base SFT model.

### Pros and cons

| Technique | Pros | Cons |
| --- | --- | --- |
| SFT on edited examples | teaches a consistent format and voice | can reproduce source biases/errors |
| RAG/source injection | updates facts without retraining | retrieval failures still cause omissions |
| DPO pairs | sharpens explicit editorial preferences | subjective pairs can encode annotator taste |
| Structured output | easy automatic validation | can lose nuance unless schema allows uncertainty |

## Task contracts

For each task, define audience, source material, output format, allowed claims, tone, maximum length, and abstention behavior. Example: “Summarize only supplied meeting notes in five bullets. Mark unresolved decisions. Do not infer commitments.”

## Dataset design

1. Gather permitted source-to-draft pairs or create expert-reviewed synthetic examples.
2. Keep source and target together; a draft without its evidence teaches unsupported generation.
3. Label factual support, required elements, style, privacy constraints, and unacceptable claims.
4. Include revision pairs: weak draft -> editorial feedback -> revised draft.
5. Partition by author, client, or document family to prevent template leakage.

## Training and evaluation

Use SFT to establish format and voice. Add DPO-style pairs for clear preferences such as “supported vs invented claim” or “direct vs needlessly verbose.” Evaluate with:

- required-field coverage and JSON/schema validity where relevant
- citation/claim support against supplied source
- rubric ratings from blinded reviewers
- length, reading level, and redaction compliance
- refusal/clarification rate for underspecified requests

## First project

Create a meeting-note transformer that returns decisions, action items with owners, risks, and open questions in fixed JSON. The evaluator can check schema and whether every item is grounded in the provided notes; humans judge whether the extraction missed nuance.

## Milestone

Demonstrate better rubric and groundedness scores than the frozen base model, with no increase in unsupported claims on a held-out source set.
