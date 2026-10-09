"""Shared sandbox for labs 02-06: a tiny char-level GPT that learns 2-digit addition.

Sequence layout (always T=10 tokens):   a1 a2 + b1 b2 =  c_ones c_tens c_hundreds \n
The answer is written least-significant digit first (makes addition easy for small
autoregressive models). Prompt = first P=6 tokens, completion = last 4 tokens.
Every answer is machine-checkable, so the same task supports SFT, DPO, rejection
sampling and GRPO with an exact verifier.
"""
from __future__ import annotations

import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "0123456789+=\n"
STOI = {c: i for i, c in enumerate(CHARS)}
PLUS, EQ, NL = STOI["+"], STOI["="], STOI["\n"]
VOCAB, T, P = len(CHARS), 10, 6
RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")


def device() -> str:
    # CPU by default for reproducibility; set LAB_DEVICE=xpu to run on the Arc GPU.
    return os.environ.get("LAB_DEVICE", "cpu")


# ----------------------------------------------------------------------------- data
def split_pairs(n_test: int = 2000, seed: int = 0):
    """All 10,000 operand pairs, split by pair so test prompts are never seen in training."""
    pairs = [(a, b) for a in range(100) for b in range(100)]
    random.Random(seed).shuffle(pairs)
    as_t = lambda ps: torch.tensor(ps, dtype=torch.long)
    return as_t(pairs[n_test:]), as_t(pairs[:n_test])


def encode(ab: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """ab: [N,2] operands, c: [N] claimed answers -> [N,T] token ids (answer reversed)."""
    a, b = ab[:, 0], ab[:, 1]
    cols = [a // 10, a % 10, torch.full_like(a, PLUS), b // 10, b % 10, torch.full_like(a, EQ),
            c % 10, (c // 10) % 10, (c // 100) % 10, torch.full_like(a, NL)]
    return torch.stack(cols, dim=1)


def prompts(ab: torch.Tensor) -> torch.Tensor:
    return encode(ab, ab.sum(1))[:, :P]


def corpus(ab: torch.Tensor, wrong_prob: float, gen: torch.Generator) -> torch.Tensor:
    """Noisy 'pretraining' corpus: with prob wrong_prob the stated answer is random."""
    c = ab.sum(1)
    noisy = torch.randint(0, 199, c.shape, generator=gen)
    c = torch.where(torch.rand(c.shape, generator=gen) < wrong_prob, noisy, c)
    return encode(ab, c)


def decode_answer(seqs: torch.Tensor):
    """Return (value, well_formed) for completions in seqs[:, P:]."""
    d = seqs[:, P:P + 3]
    ok = (d < 10).all(1) & (seqs[:, P + 3] == NL)
    value = d[:, 0] + 10 * d[:, 1] + 100 * d[:, 2]
    return value, ok


def reward(seqs: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Verifier: 1.0 iff the completion is well formed AND the sum is correct."""
    value, ok = decode_answer(seqs)
    return (ok & (value == ab.sum(1))).float()


# ----------------------------------------------------------------------------- model
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, L, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, L, D))
        return x + self.down(F.gelu(self.up(self.n2(x))))


class GPT(nn.Module):
    def __init__(self, d: int = 96, layers: int = 3, heads: int = 4):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads)
        self.emb, self.pos = nn.Embedding(VOCAB, d), nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm, self.head = RMSNorm(d), nn.Linear(d, VOCAB, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


def n_params(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# ----------------------------------------------------------------------------- LoRA
class LoRALinear(nn.Module):
    """W' = W + (alpha/r) * B @ A ; W frozen, A,B trainable, B starts at 0 (identity at init)."""

    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale

    def merged_linear(self) -> nn.Linear:
        self.base.weight.data += self.scale * (self.B @ self.A)
        return self.base


def add_lora(model: GPT, r: int = 8, alpha: float = 16, targets=("qkv", "proj", "up", "down")) -> GPT:
    for p in model.parameters():
        p.requires_grad_(False)
    for blk in model.blocks:
        for name in targets:
            setattr(blk, name, LoRALinear(getattr(blk, name), r, alpha))
    return model


def merge_lora(model: GPT) -> GPT:
    for blk in model.blocks:
        for name in ("qkv", "proj", "up", "down"):
            m = getattr(blk, name)
            if isinstance(m, LoRALinear):
                setattr(blk, name, m.merged_linear())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


# ----------------------------------------------------------------------------- sampling / scoring
@torch.no_grad()
def generate(model, prompt_ids: torch.Tensor, greedy: bool = False, temperature: float = 1.0):
    x = prompt_ids
    for _ in range(T - P):
        logits = model(x)[:, -1].float()
        if greedy:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, -1), 1)
        x = torch.cat([x, nxt], 1)
    return x


def completion_logprobs(model, seqs: torch.Tensor) -> torch.Tensor:
    """Per-token log p(token | prefix) for the 4 completion tokens -> [B, 4]."""
    logp = F.log_softmax(model(seqs[:, :-1]).float(), -1)
    tok = logp.gather(-1, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    return tok[:, P - 1:]


@torch.no_grad()
def accuracy(model, ab: torch.Tensor, greedy: bool = True, n: int = 1) -> float:
    model.eval()
    ab = ab.to(next(model.parameters()).device)
    accs = [reward(generate(model, prompts(ab), greedy=greedy), ab).mean().item() for _ in range(n)]
    return sum(accs) / len(accs)


# ----------------------------------------------------------------------------- training
def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warmup then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))


def next_token_loss(model, seqs: torch.Tensor, loss_from: int = 0) -> torch.Tensor:
    """Mean cross-entropy over target positions >= loss_from (loss_from=P masks the prompt)."""
    logits = model(seqs[:, :-1]).float()
    ce = F.cross_entropy(logits.reshape(-1, VOCAB), seqs[:, 1:].reshape(-1), reduction="none")
    ce = ce.view(seqs.shape[0], -1)[:, max(loss_from - 1, 0):]
    return ce.mean()


def fit(model, make_batch, steps: int, lr: float, loss_from: int = 0, wd: float = 0.0,
        warmup: int = 50, log_every: int = 0, params=None, tag: str = "") -> list[float]:
    """Generic SFT loop. make_batch(step) -> [B,T] token tensor on the model's device."""
    params = [p for p in (params or model.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    history = []
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr, warmup)
        model.train()
        loss = next_token_loss(model, make_batch(step), loss_from)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history.append(loss.item())
        if log_every and (step % log_every == 0 or step == steps - 1):
            print(f"  {tag} step {step:5d}  loss {loss.item():.4f}")
    return history


def save(model, name: str) -> None:
    os.makedirs(RUNS, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict()}, os.path.join(RUNS, name))


def load(name: str) -> GPT:
    blob = torch.load(os.path.join(RUNS, name), map_location=device())
    model = GPT(**blob["cfg"]).to(device())
    model.load_state_dict(blob["state"])
    return model


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)
