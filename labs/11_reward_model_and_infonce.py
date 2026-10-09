# %% [markdown]
# # Lab 11 - Reward models (Bradley-Terry) and InfoNCE retrieval
#
# Companion docs: `06-preference-optimization.md`, `13-other-use-cases.md`. Light PyTorch, CPU, seconds.
#
# Two losses that recur everywhere:
#
# **Bradley-Terry reward model.** Humans (or judges) say which of two answers is better. Model
# $P(y_w\succ y_l\mid x)=\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$ and minimise
# $\mathcal{L}_{RM}=-\log\sigma\big(r_\phi(x,y_w)-r_\phi(x,y_l)\big)$. This is exactly the classifier DPO hides inside its loss.
#
# **InfoNCE (contrastive).** For a query $q$ with positive $d^+$ and negatives $d^-_j$:
# $\mathcal{L}=-\log\frac{e^{s(q,d^+)/\tau}}{e^{s(q,d^+)/\tau}+\sum_j e^{s(q,d_j^-)/\tau}}$ - a softmax classification over documents.
# It trains embedding models and rerankers for RAG.

# %%
import torch
import torch.nn.functional as F

torch.manual_seed(0)

# %% [markdown]
# ## 1. Recover a hidden "true reward" from noisy pairwise preferences
# Each answer is a feature vector; the true reward is linear. Labels are sampled from Bradley-Terry with the true reward,
# so some labels are "wrong" - like real annotators.

# %%
d, n = 8, 4000
w_true = torch.randn(d)
xa, xb = torch.randn(n, d), torch.randn(n, d)
p_a_wins = torch.sigmoid((xa - xb) @ w_true)
a_wins = torch.bernoulli(p_a_wins)
chosen = torch.where(a_wins[:, None] == 1, xa, xb)
rejected = torch.where(a_wins[:, None] == 1, xb, xa)

w = torch.zeros(d, requires_grad=True)
opt = torch.optim.Adam([w], lr=0.05)
for step in range(400):
    loss = -F.logsigmoid((chosen - rejected) @ w).mean()
    opt.zero_grad(); loss.backward(); opt.step()
cos = F.cosine_similarity(w.detach(), w_true, dim=0).item()
acc = (((chosen - rejected) @ w) > 0).float().mean().item()
print(f"final loss {loss.item():.3f}; cosine(w, w_true) = {cos:.3f}; preference accuracy {acc:.3f}")
print("The ceiling on preference accuracy is below 100% because labels are noisy - expect the same in real RM data (~70-75%).")

# %% [markdown]
# ## 2. Reward over-optimisation (Goodhart)
# The learned reward is only a proxy. If you optimise hard against it, the *true* reward eventually drops.
# Simulate by searching for the candidate that maximises the proxy among $N$ random candidates (best-of-$N$).

# %%
def best_of_n(N, trials=300):
    true_scores = []
    for _ in range(trials):
        cand = torch.randn(N, d)
        proxy = cand @ (w.detach() + 0.6 * torch.randn(d))          # imperfect reward model
        true_scores.append((cand[proxy.argmax()] @ w_true).item())
    return sum(true_scores) / len(true_scores)


for N in (1, 4, 16, 64, 256):
    print(f"best-of-{N:<3d}: mean TRUE reward = {best_of_n(N):+.3f}")
print("True reward rises with N but with diminishing returns; with a worse proxy it can turn around. Always evaluate on held-out *true* metrics.")

# %% [markdown]
# ## 3. InfoNCE: train a tiny retriever
# Synthetic "documents" are bags of topic words; a query is a noisy subset of one document. We learn embeddings with
# in-batch negatives and measure Recall@1 before and after training.

# %%
vocab, n_docs, dim = 200, 500, 32
docs = (torch.rand(n_docs, vocab) < 0.05).float()
queries = docs * (torch.rand_like(docs) < 0.5).float()            # keep half of each doc's words


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(vocab, dim, bias=False)

    def forward(self, x):
        return F.normalize(self.lin(x), dim=-1)


enc = Encoder()


def recall_at(k):
    with torch.no_grad():
        s = enc(queries) @ enc(docs).T
        return (s.topk(k, dim=1).indices == torch.arange(n_docs)[:, None]).any(1).float().mean().item()


print(f"before training: Recall@1={recall_at(1):.3f}  Recall@5={recall_at(5):.3f}")
opt = torch.optim.Adam(enc.parameters(), lr=0.02)
tau = 0.05
for step in range(300):
    idx = torch.randint(0, n_docs, (128,))
    q, dpos = enc(queries[idx]), enc(docs[idx])
    logits = q @ dpos.T / tau                                      # other docs in the batch are the negatives
    loss = F.cross_entropy(logits, torch.arange(len(idx)))
    opt.zero_grad(); loss.backward(); opt.step()
print(f"after  training: Recall@1={recall_at(1):.3f}  Recall@5={recall_at(5):.3f}  (final loss {loss.item():.3f})")

# %% [markdown]
# ## Exercises
# 1. Randomly flip 30% of preference labels in section 1. How does `cos(w, w_true)` degrade? (Data noise vs reward-model quality.)
# 2. Change `tau` to 1.0 and 0.01. Which setting trains fastest? What does a low temperature do to hard negatives?
# 3. Add *hard negatives* (docs sharing many words with the query) to the batch. Does Recall@1 improve more than with random negatives?
