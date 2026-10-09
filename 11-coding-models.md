# 11. Coding Models

**Goal:** specialise a small model for code: completion, fill-in-the-middle, bug fixing, test writing and small repository tasks, using executable feedback instead of guesses.
**Lab:** `labs/10_code_sandbox_and_passk`. Builds on chapters 4, 7, 8.

## 11.1 Why code is the friendliest domain for RL

Code has a built-in verifier: **run it against tests**. That gives exact, cheap, automatic rewards - the ideal setting for rejection sampling and RLVR. The hard parts are *isolation* (running untrusted output), *test quality* (incomplete tests can be gamed), and *context* (real repositories do not fit in a small model's window).

## 11.2 Task types (increasing difficulty)

| Task | Input -> output | Verifier |
| --- | --- | --- |
| Function from docstring | signature + docstring -> body | unit tests |
| Fill-in-the-middle (FIM) | prefix + suffix -> middle | exact match / tests |
| Bug fix | buggy function + failing test -> fixed function | tests pass |
| Test generation | function -> tests | mutation score / coverage / catch injected bugs |
| Code explanation / review | code -> text | rubric, judge |
| Repository issue fix | issue + repo -> patch | project test suite (SWE-bench style) |
| Agentic coding | tools: read/search/run/edit | tests + policy checks |

**Start at the top of the table.** Function-level tasks run on 0.5-4B models; repo-level tasks usually need retrieval, tools and larger models.

## 11.3 Model selection and data

* Prefer code-specialised or strong general small models (e.g. Qwen2.5-Coder, StarCoder2, DeepSeek-Coder families; check for newer releases and each license).
* **FIM training format:** split a file into prefix, middle, suffix and reorder with sentinel tokens (lab 10 section 5); names of sentinels are model-specific - follow the model card.
* **Data sources:** permissively licensed code (record licenses; beware copyleft), your own repositories (with permission), synthetic problems with tests, **OSS-Instruct**-style generation (Magicoder, Wei et al. 2023: seed a teacher with real code snippets to produce diverse tasks), commit histories as (before, instruction, after) triples, and mutation-generated bugs (below).
* **Split by repository and time.** Near-duplicate code is everywhere; deduplicate and decontaminate against your evaluation sets and public benchmarks.

### Mutation-based bug-fix dataset (programmatic, exactly verifiable)
1. Take working functions with tests (from your repos, or generate them).
2. Apply mutations: flip operators (`<`<->`<=`), off-by-one, wrong variable, removed guard, swapped arguments, changed constant.
3. Keep mutants that **fail** at least one test.
4. Prompt: buggy code + failing test output; target: original. The verifier is the test suite, and the original is the reference fix.

## 11.4 Patch formats matter for small models

| Output format | Pros | Cons |
| --- | --- | --- |
| Whole function / file | easiest to generate and verify | long outputs for big files |
| Search/replace blocks | compact; no line numbers | needs exact-match application; fails if search text is wrong |
| Unified diff | standard | small models often get hunk headers/line numbers wrong (commonly observed - test on your model) |
| Tool calls (edit file at symbol) | structured | needs an agent loop |

Measure **patch apply rate** separately from **test pass rate**; fix the first before blaming reasoning.

## 11.5 Metrics

* **$\text{pass@}k$** with the unbiased estimator (lab 05/10): functional correctness.
* Patch apply rate, tests pass, regression tests still pass, changed-lines/files, time and tokens per success.
* Benchmarks: HumanEval (Chen et al. 2021) and MBPP (function level; saturated and contaminated for many models), **LiveCodeBench** (contamination-aware, time-stamped problems), **BigCodeBench**, **SWE-bench** / Verified (Jimenez et al. 2023) for issue resolution, and SWE-agent (Yang et al. 2024) for the agent-computer interface idea. Always add a **private** benchmark built from your own code.
* Non-functional checks: static analysis/lint, type checks, security scan, license scan, maintainability review.

## 11.6 Training recipe

1. **Baseline** base/instruct model with greedy and $n=8$ sampling at $T=0.8$: pass@1, pass@8, apply rate.
2. **SFT** (LoRA) on 2-20k high-quality examples in the exact prompt/patch format.
3. **Rejection sampling:** sample $k=8$ per task, run tests in the sandbox, keep passing solutions (prefer shorter/cleaner), retrain.
4. **RLVR (GRPO):** reward = 1 if all *hidden* tests pass and static policy checks pass; optional small bonus for passing more tests (partial credit); penalty for policy violations. Keep completions <= 512-1024 tokens; mask truncated completions.
5. **Hard-case mining:** collect failures, categorise (syntax, API misuse, edge cases, off-by-one), generate targeted tasks.
6. **Evaluate** on held-out repositories; inspect 20 accepted patches for overfitting to tests.

## 11.7 Reward hacking in code (test for each)

| Hack | Countermeasure |
| --- | --- |
| Special-casing visible test inputs | hidden tests; randomised/property-based tests |
| Editing or deleting test files / `sys.exit(0)` | run tests from a read-only copy; check exit status + junit report; forbid writes outside the solution file |
| Exploiting timeouts or environment | resource limits; deterministic seeds |
| Printing the expected string | compare returned values, not stdout, where possible |
| Import tricks / monkeypatching the test harness | isolate interpreter (`-I`), policy scan, separate processes |
| Network/file access | no network, read-only filesystem, container |

## 11.8 Execution safety

Model-written code is **untrusted**. A subprocess with a timeout is for learning only (lab 10 says so). For real pipelines: container or micro-VM, no network, non-root user, CPU/memory/disk/time limits, read-only mounts, ephemeral filesystem, and kill-on-timeout. Never run it on a machine with your credentials.

## 11.9 Context and memory on 12 GB

Repository tasks are limited by context length, not parameters: use **retrieval** (BM25/embedding over files/symbols) to place 2-5 relevant files in the prompt; include the failing test and traceback; drop unrelated code. Long contexts multiply activation memory; start with 4-8k tokens.

## 11.10 Licensing and security of generated code

Training on code carries license obligations (copyleft, attribution). Generated code can reproduce training snippets verbatim and can contain vulnerabilities (injection, unsafe deserialisation, weak crypto). Add security linting to your evaluation and keep a human reviewing anything that ships.

## 11.11 Pros and cons

| Approach | Pros | Cons |
| --- | --- | --- |
| FIM SFT | great for editors/IDEs; cheap | not for chat-style tasks alone |
| Instruction SFT | broad behaviour | quality depends on data; imitation of buggy samples |
| Rejection sampling | verified data, simple | limited by pass@k |
| RLVR | optimises pass rate directly | sampling cost; test gaming; sandbox needed |
| Agentic coding | handles repos | long contexts, many failure modes; larger models needed |

## Checkpoint

* Sandbox + hidden tests + pass@k implemented; apply rate and pass rate reported separately.
* A mutation-generated bug-fix dataset with exact verification.
* A list of reward hacks you tested against and what you observed.
