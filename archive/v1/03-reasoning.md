# 3. Reasoning Models

## Goal

Make a small model more reliable at problems with checkable outcomes: arithmetic, rules, plans, database queries, and tool calls. Reasoning is not a decorative long explanation; it is a process that improves verified task success.

## Start with verifiable tasks

Pick outcomes a program can check: a numerical answer, unit test, SQL execution, symbolic proof checker, JSON schema, or known action result. This enables cheap feedback and is much more suitable for small-model RL than subjective “sounds smart” ratings.

## Lesson: reasoning is search plus feedback

Run `python3 labs/03_verifier_and_rejection_sampling.py`. It samples arithmetic candidates, validates them with a deterministic checker, and retains only successful trajectories. The important idea is not arithmetic: the verifier changes noisy model outputs into trustworthy training examples.

Given a question $q$, generate $k$ candidates $y_1, \ldots, y_k$. A verifier $V(q,y)$ returns 1 for success and 0 otherwise. Rejection sampling retains:

$$D_{good} = \{(q,y_i) : V(q,y_i)=1\}$$

Then SFT on $D_{good}$ increases the probability of successful forms. This works only when the verifier checks the outcome well enough. A weak verifier trains loopholes.

### Step by step: verified reasoning

1. Pick a task with an executable checker: arithmetic, read-only SQL, JSON transformation, or unit-tested code.
2. Implement the checker before collecting training traces. Add tests for malformed output and unsafe actions.
3. Measure base-model pass@1 and pass@4. Keep the questions and checker versions fixed.
4. Generate 4-8 candidate answers per training prompt at a nonzero temperature.
5. Parse and verify each output. Store prompt, candidate, parse result, and verifier result.
6. Fine-tune on only the verified candidates, retaining a separate untouched evaluation set.
7. Re-run the original benchmark and classify each failure: wrong plan, wrong tool, parse failure, or unsupported knowledge.

### Pros and cons

| Technique | Pros | Cons |
| --- | --- | --- |
| Long rationales | may reveal intermediate errors | expensive; model can imitate verbosity rather than solve |
| Tool use | externalizes exact calculation/search | tool outputs and errors become another training distribution |
| Rejection sampling | simple and stable; no RL loop | needs a nonzero base success rate and a strong verifier |
| Outcome RL | directly optimizes success | can exploit reward mistakes and is harder to stabilize |

## Progressive curriculum

1. **Prompt baseline**: few-shot examples, calculator/code/search tools, and a required final-answer format.
2. **SFT on solved examples**: train concise scratchpads or tool trajectories only when they improve correctness.
3. **Distillation**: obtain verified solutions from a stronger teacher or solver, filter them, and train the small student.
4. **Rejection sampling**: generate several candidate solutions, retain only verifier-passing ones, and fine-tune on the retained set.
5. **Outcome-RL**: optimize directly against a deterministic reward once the SFT policy can occasionally succeed.

## Design a verifier first

```text
input -> model candidate -> parser -> deterministic verifier -> reward
```

For a math task, parse the boxed/final answer and compare it to the answer key. For code, run a restricted test suite. For tool plans, execute them in a sandbox. Reward both correctness and format compliance; penalize unsafe or invalid calls.

## Reasoning data rules

- Store the final answer separately from optional rationale.
- Check every generated rationale with the verifier. Fluent incorrect rationales poison training.
- Prefer short, useful traces; small models have limited context and may imitate verbosity instead of solving.
- Keep tool outputs in training examples exactly as the agent will see them at inference time.

## Evaluation

Report exact-match/pass@k, verifier pass rate, format-validity rate, tool-call error rate, latency, and average tokens. Inspect failure clusters: parsing, knowledge, multi-step planning, tool selection, or execution error. Each cluster needs a different repair.

## First project

Build a “data analyst” assistant for a toy SQLite database: it receives a question, emits a read-only SQL query, gets the result, and returns a short answer. Verify SQL syntax, block mutations, execute in an isolated database, and score answer correctness. This is reasoning plus agent behavior without uncontrolled computer access.

## Milestone

Demonstrate that a verifier-filtered SFT model improves held-out verifier pass rate over both the base model and an unfiltered-SFT control.
