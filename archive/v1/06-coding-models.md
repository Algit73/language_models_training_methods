# 6. Coding Models

## Goal

Train a compact coding assistant that navigates a bounded repository, makes a patch, and proves it with tests.

## Pick a constrained task

Good first tasks are docstring-to-function, bug fix in a small repository, unit-test generation, code review classification, or JSON-to-code transformation. Avoid training on a broad, unlicensed scrape of public code. Preserve licenses and provenance for every source.

## Training examples

Use realistic context and machine-checkable outcomes:

```json
{"messages":[{"role":"user","content":"Fix the failing test.\nFILE: math_utils.py\n..."},{"role":"assistant","content":"<patch>...</patch>"}],"tests":"pytest tests/test_math_utils.py"}
```

Include repository context, error output, a patch format, and the expected test command. Train response masking so the model learns to emit the patch, not repeat the prompt.

## Lesson: code training is verified generation

Run `python3 labs/06_patch_evaluator.py`. It creates a disposable Python module, applies a controlled patch, and runs a unit test in a temporary directory. The same structure powers a code-agent evaluator: isolate, apply, test, and discard.

For candidate patch $p$ on issue $x$, define a reward only after isolation:

$$r(x,p)=\mathbb{1}[\text{patch applies}]\,\mathbb{1}[\text{allowed tests pass}] - \lambda\,\mathbb{1}[\text{policy violation}]$$

Do not reward raw test success without a policy check. A candidate could delete a test, change unrelated files, or introduce a security regression.

### Step by step: a coding specialization

1. Select small repositories with explicit licenses and create issues from historical commits or hand-authored bugs.
2. For every issue, store the pre-change repository, failing test, allowed command, expected patch, and policy constraints.
3. Split by repository or commit family so near-identical code does not occur in train and test.
4. Benchmark the base model's patch syntax, apply rate, test pass rate, and changed-file count.
5. SFT on issue-to-patch pairs. Keep context to relevant files found by retrieval.
6. Generate $k$ candidates in fresh temporary worktrees, run allowlisted tests with timeouts, and retain logs.
7. Use passing, policy-compliant patches for verified fine-tuning; manually review the held-out wins.

### Pros and cons

| Approach | Pros | Cons |
| --- | --- | --- |
| Patch-only SFT | simple and cheap | depends on supplied context being sufficient |
| Tool-using code agent | can search/test iteratively | longer contexts, extra failure modes, must sandbox |
| Test-based reward | objective and scalable | tests can be incomplete or gamed |

## Development loop

1. Build a tiny benchmark of issues held out by repository and commit.
2. Measure the base model: patch parses, applies cleanly, tests pass, and no unrelated files changed.
3. SFT a LoRA adapter on high-quality issue-to-patch examples.
4. Generate $k$ patches, apply each in a temporary worktree/container, and run restricted tests.
5. Use passing patches as verified data for rejection-sampling SFT or reward signals.
6. Add tools gradually: file search, read file, run selected tests, then patch. Every tool needs a timeout and allowlist.

## Metrics

Use patch-apply rate, test pass@1/pass@k, regression-test rate, changed-lines count, syntax validity, and tool-call cost. Manual review should check security, scope, and maintainability; a passing test is not complete evidence.

## B580 strategy

Use a 1.5B-4B code/instruct model, QLoRA, files trimmed to relevant context, and targeted tests. Context length is a chief constraint: a retrieval step that finds the right 2-5 files often outperforms blindly increasing the model size.

## Milestone

Produce a model card and benchmark report showing a tested improvement over the base model on held-out issues, including failures that still require a human developer.
