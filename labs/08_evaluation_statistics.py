# %% [markdown]
# # Lab 08 - Evaluation statistics: confidence intervals, paired tests, contamination
#
# Companion doc: `14-evaluation.md`. Pure Python (+ optional numpy). No checkpoint needed.
#
# Most "model A beats model B by 2 points" claims from a few hundred examples are noise. This lab
# gives you the tools to know when a difference is real.
#
# * Accuracy on $n$ items is a binomial proportion: standard error $\sqrt{\hat p(1-\hat p)/n}$.
# * When two models are run on the **same** items, use a **paired** comparison (much more powerful).
# * Always check for **contamination**: test items that leaked into training data.

# %%
import math
import random

random.seed(0)

# %% [markdown]
# ## 1. How wide is the error bar?

# %%
def wilson(k, n, z=1.96):
    """Wilson score interval: better than +-1.96*SE for small n or extreme p."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


for n in (50, 200, 1000, 5000):
    lo, hi = wilson(int(0.6 * n), n)
    print(f"n={n:5d}  acc=0.60  95% CI = [{lo:.3f}, {hi:.3f}]  (width {hi - lo:.3f})")

# %% [markdown]
# ## 2. Sample size needed to detect an improvement
# To distinguish accuracy $p_1$ from $p_2=p_1+\delta$ with two *independent* samples at 95% confidence / 80% power:
# $n\approx 2\,\frac{(z_{\alpha/2}+z_\beta)^2\,\bar p(1-\bar p)}{\delta^2}$ per model, with $z_{\alpha/2}=1.96,\ z_\beta=0.84$.

# %%
def n_needed(p=0.6, delta=0.02):
    return math.ceil(2 * (1.96 + 0.84) ** 2 * p * (1 - p) / delta**2)


for d in (0.10, 0.05, 0.02, 0.01):
    print(f"to detect +{d * 100:.0f} points at p~0.6 you need about {n_needed(0.6, d):6d} items per model (independent test)")

# %% [markdown]
# ## 3. Paired bootstrap: same items, two models
# Simulate model A and B whose correctness is *correlated* (hard items are hard for both).

# %%
def simulate(n=500, pa=0.62, pb=0.66, corr_strength=0.8):
    a, b = [], []
    for _ in range(n):
        difficulty = random.random()                      # shared latent difficulty
        ta = pa + (0.5 - difficulty) * corr_strength
        tb = pb + (0.5 - difficulty) * corr_strength
        a.append(int(random.random() < min(max(ta, 0), 1)))
        b.append(int(random.random() < min(max(tb, 0), 1)))
    return a, b


def paired_bootstrap(a, b, iters=5000):
    n = len(a)
    diffs = []
    for _ in range(iters):
        idx = [random.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


a, b = simulate()
lo, hi = paired_bootstrap(a, b)
print(f"A={sum(a) / len(a):.3f}  B={sum(b) / len(b):.3f}  paired 95% CI for (B-A): [{lo:+.3f}, {hi:+.3f}]")
print("Significant" if lo > 0 or hi < 0 else "NOT distinguishable from zero")

# %% [markdown]
# ## 4. McNemar's exact-style view
# Only items where the models **disagree** carry information: $n_{10}$ (A right, B wrong) and $n_{01}$ (A wrong, B right).
# Under "no difference", $n_{01}\sim\mathrm{Binomial}(n_{01}+n_{10},\tfrac12)$.

# %%
def mcnemar_p(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    m = n10 + n01
    if m == 0:
        return 1.0
    k = min(n10, n01)
    p = sum(math.comb(m, i) for i in range(0, k + 1)) / 2**m
    return min(1.0, 2 * p)


print(f"discordant pairs: {sum(x != y for x, y in zip(a, b))};  two-sided McNemar p = {mcnemar_p(a, b):.4f}")

# %% [markdown]
# ## 5. Contamination check: n-gram overlap
# If a test item (or its answer) appears in your training set, the score is meaningless. A cheap, conservative check:
# flag any test item sharing a long word n-gram with any training document.

# %%
def ngrams(text, n=8):
    w = text.lower().split()
    return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}


train_docs = [
    "to install the package run pip install foo and then import foo in your script before anything else happens",
    "the quick brown fox jumps over the lazy dog while the cat watches quietly from the wall nearby",
]
test_items = [
    "run pip install foo and then import foo in your script before anything else happens today",   # near-duplicate
    "explain how gradient accumulation reduces peak memory during fine tuning of a language model",
]
train_grams = set().union(*(ngrams(d) for d in train_docs))
for t in test_items:
    hit = ngrams(t) & train_grams
    print(f"{'CONTAMINATED' if hit else 'clean':12s} overlaps={len(hit)}  :: {t[:60]}")

# %% [markdown]
# ## 6. Report template (copy into every experiment)
# ```
# model / revision / adapter:        ...
# dataset + split + n:               ...
# decoding (temp, top_p, max_tokens, seed, #samples): ...
# metric +- 95% CI:                  ...
# paired comparison vs baseline:     delta [CI], McNemar p
# contamination check performed:     yes/no (method)
# 20 inspected failures:             link / categories
# ```
#
# ## Exercises
# 1. For `n=200`, what is the smallest paired improvement your data can reliably detect? Try different `corr_strength` values. What does high correlation buy you?
# 2. Change `pa, pb` to be equal. Run the bootstrap 20 times. How often does it (wrongly) call the difference significant? (That is your false positive rate.)
# 3. Extend `ngrams` to character n-grams and test paraphrase-robust matching. What do you catch, and what slips through?
