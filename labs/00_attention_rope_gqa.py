# %% [markdown]
# # Lab 00 - Attention, RoPE and GQA from scratch
#
# Goal: see that the core of a modern LLM is a few lines of linear algebra, and verify each piece
# against the library implementation. Pure PyTorch, runs on CPU in seconds.
#
# Companion doc: `01-foundations-transformer-math.md`.
#
# **Attention** for one head: $\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V$,
# where $M$ is $-\infty$ above the diagonal (causal mask: token $t$ may not look at $t+1,\dots$).

# %%
import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)

# %% [markdown]
# ## 1. Scaled dot-product attention by hand
# Compare our explicit version with `F.scaled_dot_product_attention` (what real models call).

# %%
def attention(q, k, v, causal=True):
    # q,k,v: [batch, heads, length, head_dim]
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)          # [B,H,L,L]
    if causal:
        L = q.shape[-2]
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ v


B, H, L, D = 2, 4, 8, 16
q, k, v = (torch.randn(B, H, L, D) for _ in range(3))
ours = attention(q, k, v)
ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
print("max abs difference vs library:", (ours - ref).abs().max().item())

# %% [markdown]
# ## 2. Why divide by $\sqrt{d_k}$?
# If $q,k$ have unit-variance entries, $q\cdot k$ has variance $d_k$. Large scores push softmax into
# a near one-hot regime where gradients vanish. Dividing by $\sqrt{d_k}$ restores unit variance.

# %%
for d in (16, 256, 4096):
    q1, k1 = torch.randn(10000, d), torch.randn(10000, d)
    raw = (q1 * k1).sum(-1)
    print(f"d_k={d:5d}  std(raw)={raw.std():7.2f}  std(scaled)={(raw / math.sqrt(d)).std():5.2f}")

row = torch.randn(64) * 1.0
for scale in (1.0, 8.0):
    p = F.softmax(row * scale, -1)
    print(f"logit scale {scale}: max prob {p.max():.3f}, entropy {-(p * p.log()).sum():.3f}")

# %% [markdown]
# ## 3. Rotary position embeddings (RoPE)
# Rotate each pair of channels $(x_{2i},x_{2i+1})$ by angle $m\theta_i$ with $\theta_i=10000^{-2i/d}$,
# where $m$ is the token position. Key property: $\langle R_m q, R_n k\rangle$ depends only on $m-n$,
# so attention sees *relative* distance without a learned position table.

# %%
def rope_tables(head_dim, max_pos, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    ang = torch.outer(torch.arange(max_pos).float(), inv)       # [pos, d/2]
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[: x.shape[-2]], sin[: x.shape[-2]]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out


d, n_pos = 32, 16
cos, sin = rope_tables(d, n_pos)
qv, kv = torch.randn(d), torch.randn(d)
Q = apply_rope(qv.expand(n_pos, d).clone(), cos, sin)    # same vector placed at every position
K = apply_rope(kv.expand(n_pos, d).clone(), cos, sin)
S = Q @ K.T
print("score(m=5,n=2) :", S[5, 2].item())
print("score(m=9,n=6) :", S[9, 6].item(), " <- same offset (3), same score")
print("score(m=9,n=2) :", S[9, 2].item(), " <- different offset")

# %% [markdown]
# ## 4. Grouped-query attention (GQA) and the KV cache
# With $h$ query heads but only $g$ key/value heads ($g \mid h$), each K/V head is shared by $h/g$
# query heads. The KV cache stores $2\cdot \text{layers}\cdot g\cdot d_h\cdot \text{seq}\cdot \text{batch}\cdot \text{bytes}$,
# so GQA shrinks inference memory by $h/g$ with small quality loss. This matters on a 12 GB card.

# %%
def gqa(q, k, v):
    # q: [B,h,L,d], k/v: [B,g,L,d]
    h, g = q.shape[1], k.shape[1]
    k = k.repeat_interleave(h // g, dim=1)
    v = v.repeat_interleave(h // g, dim=1)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


q = torch.randn(1, 16, 8, 64)
k, v = torch.randn(1, 4, 8, 64), torch.randn(1, 4, 8, 64)
print("GQA output shape:", tuple(gqa(q, k, v).shape))


def kv_cache_gib(layers, kv_heads, head_dim, seq, batch=1, bytes_per=2):
    return 2 * layers * kv_heads * head_dim * seq * batch * bytes_per / 1024**3


print("28 layers, head_dim 128, seq 8192, bf16:")
print("  MHA (16 kv heads):", round(kv_cache_gib(28, 16, 128, 8192), 2), "GiB")
print("  GQA ( 8 kv heads):", round(kv_cache_gib(28, 8, 128, 8192), 2), "GiB")

# %% [markdown]
# ## 5. RMSNorm and SwiGLU
# * RMSNorm: $y = \frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g$ (LayerNorm without mean-centering or bias).
# * SwiGLU MLP: $y = W_d\big(\mathrm{SiLU}(W_g x)\odot W_u x\big)$ - three matrices instead of two.

# %%
x = torch.randn(4, 64) * 5 + 3
rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
print("rows have unit RMS after RMSNorm:", rms.pow(2).mean(-1).sqrt())

d_model, d_ff = 1024, 3072
swiglu_params = 3 * d_model * d_ff
classic_params = 2 * d_model * (4 * d_model)
print("SwiGLU MLP params :", swiglu_params)
print("classic 4x MLP    :", classic_params)

# %% [markdown]
# ## Exercises
# 1. Remove the causal mask and explain why a language model trained that way "cheats".
# 2. Change `base` in `rope_tables` to 500 and 1_000_000. Which frequencies change most? (This is the idea behind RoPE base scaling for longer context.)
# 3. Compute the KV cache for your favourite model config at 32k tokens. Does it fit next to the weights in 12 GiB?
