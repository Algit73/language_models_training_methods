# %% [markdown]
# # Lab 10 - A code-execution sandbox and pass@k with real tests
#
# Companion doc: `11-coding-models.md`. Pure standard library.
#
# Verified code generation needs three guarantees: **isolation** (candidate code cannot harm the machine),
# **determinism** (same candidate -> same result) and **a timeout**. This lab implements a minimal harness you can
# later plug into rejection sampling or a GRPO reward.
#
# > Security note: a subprocess with a timeout is *not* a real sandbox. For untrusted model output use a container
# > or VM without network access and with CPU/memory/disk limits. This lab is for trusted, local learning only.

# %%
import ast
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

# %% [markdown]
# ## 1. Policy check before execution
# Reject obviously disallowed constructs with the AST. This is a *first* filter, never the only defence.

# %%
FORBIDDEN_IMPORTS = {"os", "sys", "subprocess", "socket", "shutil", "pathlib", "ctypes", "requests"}
FORBIDDEN_CALLS = {"open", "exec", "eval", "compile", "__import__", "input"}


def policy_violations(source: str) -> list[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"syntax error: {e.msg}"]
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bad += [f"import {a.name}" for a in node.names if a.name.split(".")[0] in FORBIDDEN_IMPORTS]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] in FORBIDDEN_IMPORTS:
            bad.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            bad.append(f"call {node.func.id}()")
    return bad

# %% [markdown]
# ## 2. Run candidate + tests in a fresh interpreter with a timeout
# `-I` isolates the interpreter from user site-packages/env vars. We run in a temporary directory that is deleted afterwards.

# %%
def run_candidate(source: str, tests: str, timeout: float = 5.0) -> dict:
    viol = policy_violations(source)
    if viol:
        return {"status": "policy_violation", "detail": viol}
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "solution.py").write_text(source, encoding="utf-8")
        Path(tmp, "test_solution.py").write_text(tests, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", "-m", "unittest", "-q", "test_solution"], cwd=tmp,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "detail": ""}
    return {"status": "pass" if r.returncode == 0 else "fail", "detail": r.stderr[-300:]}

# %% [markdown]
# ## 3. A task with hidden tests and several candidate solutions
# Include edge cases in tests: most model bugs live there.

# %%
TESTS = textwrap.dedent('''
    import unittest
    from solution import second_largest

    class T(unittest.TestCase):
        def test_basic(self):      self.assertEqual(second_largest([1, 5, 3]), 3)
        def test_duplicates(self): self.assertEqual(second_largest([5, 5, 3]), 3)
        def test_negative(self):   self.assertEqual(second_largest([-1, -5, -3]), -3)
        def test_too_short(self):
            with self.assertRaises(ValueError): second_largest([1])
''')

CANDIDATES = {
    "correct": "def second_largest(xs):\n    u = sorted(set(xs), reverse=True)\n    if len(u) < 2: raise ValueError('need 2 distinct')\n    return u[1]\n",
    "ignores_duplicates": "def second_largest(xs):\n    if len(xs) < 2: raise ValueError('short')\n    return sorted(xs, reverse=True)[1]\n",
    "infinite_loop": "def second_largest(xs):\n    while True:\n        pass\n",
    "uses_os": "import os\ndef second_largest(xs):\n    os.system('echo hi')\n    return 0\n",
    "syntax_error": "def second_largest(xs) return 1\n",
}
results = {name: run_candidate(src, TESTS, timeout=2.0) for name, src in CANDIDATES.items()}
for name, res in results.items():
    print(f"{name:20s} -> {res['status']}")

# %% [markdown]
# ## 4. From verdicts to a reward and to pass@k
# * Binary reward for RL: `1.0` only if all tests pass and no policy violation.
# * pass@k for evaluation (unbiased estimator from lab 05).
# * Watch for *test hacking*: candidates that special-case the visible tests. Keep **hidden** tests for evaluation; rewards from
#   tests the model has seen during training will be exploited.

# %%
from math import comb


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


n = len(results)
c = sum(r["status"] == "pass" for r in results.values())
print(f"n={n} samples, c={c} correct -> pass@1={pass_at_k(n, c, 1):.2f}, pass@3={pass_at_k(n, c, 3):.2f}")


def code_reward(source: str, tests: str) -> float:
    return 1.0 if run_candidate(source, tests)["status"] == "pass" else 0.0


print("reward(correct) =", code_reward(CANDIDATES["correct"], TESTS), "| reward(ignores_duplicates) =", code_reward(CANDIDATES["ignores_duplicates"], TESTS))

# %% [markdown]
# ## 5. Fill-in-the-middle (FIM) data format
# Code models are also trained to *infill*: given prefix and suffix, produce the middle. The data transform is simple:
# split a file at two random points and reorder with sentinel tokens (names differ per model - check the model card).

# %%
def to_fim(code: str, i: int, j: int, pre="<fim_prefix>", suf="<fim_suffix>", mid="<fim_middle>") -> str:
    prefix, middle, suffix = code[:i], code[i:j], code[j:]
    return f"{pre}{prefix}{suf}{suffix}{mid}{middle}"


sample = "def add(a, b):\n    return a + b\n"
print(to_fim(sample, 15, 32))

# %% [markdown]
# ## Exercises
# 1. Add a test that a correct solution must run in under 1 s for a list of 10^6 items. Which candidates now fail?
# 2. Write a candidate that detects the test file and hard-codes the expected outputs. How would hidden tests catch it?
# 3. Extend `policy_violations` to forbid attribute access like `__class__`/`__subclasses__`. Why do blocklists remain incomplete, and what does that imply for real isolation?
