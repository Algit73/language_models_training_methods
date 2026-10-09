# 15. Merging, Quantization and Serving

**Goal:** turn a trained adapter into something you can run fast and cheaply on your B580, without silently breaking quality.
**Script:** `scripts/05_merge_and_export`.

> Serving-tool support for Intel Arc (Battlemage) on WSL2 changes often. Everything about specific runtimes below is a *pointer to verify*, not a guarantee.

## 15.1 From adapter to deployable model

| Artifact | What it is | Pros | Cons |
| --- | --- | --- | --- |
| LoRA adapter | small `A,B` matrices (tens of MB) | trivial to store/swap; many tasks per base | needs base at load; some runtimes slower unmerged |
| Merged 16-bit model | $W_0+\frac{\alpha}{r}BA$ folded in | standard model, no extra latency | one task per copy |
| Quantised model (GGUF/AWQ/GPTQ/NF4) | lower-precision weights | 2-4x smaller and often faster | accuracy loss; must re-evaluate |

**Merging steps:** load the *16-bit* base, apply the adapter, `merge_and_unload()`, save with the tokenizer and **the exact chat template and generation config**. Verify logits match adapter-on-base (script 05). Never merge into an already-quantised base and save that; you would bake quantisation error into the weights.

**Multi-adapter serving:** modern engines (e.g. vLLM, SGLang) can keep many LoRA adapters resident and batch requests across them; useful if you have several task adapters on one base.

## 15.2 Quantisation fundamentals

Uniform quantisation maps a float $w$ to an integer $q$ with scale $s$ and zero-point $z$:
$$q=\mathrm{clip}\big(\mathrm{round}(w/s)+z,\ q_{min},q_{max}\big),\qquad \hat w=s\,(q-z)$$
Error per weight is at most $s/2$; with $b$ bits and range $R$, $s=R/(2^b-1)$, so each removed bit **doubles** the error. **Block-wise scales** (per 32-128 weights) and outlier handling keep accuracy.

| Method | Idea | Notes |
| --- | --- | --- |
| **Round-to-nearest (RTN), INT8/INT4** | quantise directly | simple; 8-bit nearly lossless, 4-bit needs care |
| **GPTQ** (Frantar et al. 2022) | layer-wise second-order (Hessian-based) error compensation using calibration data | good 4-bit accuracy; calibration set matters |
| **AWQ** (Lin et al. 2023) | protect salient weight channels by activation-aware scaling | good 4-bit accuracy, fast kernels |
| **GGUF k-quants** (llama.cpp) | block-wise mixed-precision formats like Q4_K_M, Q5_K_M, Q6_K, Q8_0 | broad hardware support; CPU/GPU inference |
| **NF4 (bitsandbytes)** | normal-quantile 4-bit, for QLoRA | training/inference inside PyTorch; XPU backend exists |
| **FP8 / MXFP4 etc.** | floating low precision | hardware-dependent; check Battlemage support |
| **KV-cache quantisation** | store keys/values in 8/4-bit | cuts long-context memory; evaluate quality |

**Rules of thumb (verify on your model):** 8-bit ~ no loss; 4-bit (good method) small loss on large models, larger loss on small models (1-3B) and on long multi-step reasoning/code; 3-bit and below degrade rapidly. **Re-run your evaluation after quantising** (script 04), especially agent and reasoning tasks; compare at equal *memory*, not equal parameter count (a 4-bit 4B model may beat a 16-bit 1.7B model).

## 15.3 Inference performance math

* **Decode speed** (single stream) $\lesssim\frac{\text{memory bandwidth}}{\text{bytes of weights read per token}}$ (+ KV cache reads). Quantising weights speeds decoding roughly in proportion to bytes saved if kernels are efficient.
* **Prefill** is compute bound (process the prompt in parallel): time $\approx\frac{2N\cdot\text{prompt tokens}}{\text{achieved FLOPS}}$.
* **KV cache** per sequence $=2\,L\,g\,d_h\,s\cdot\text{bytes}$; concurrency $\times$ cache + weights must fit in VRAM.
* **Latency targets:** time-to-first-token (prefill), inter-token latency (decode), total = TTFT + tokens x ITL.
* **Batching** raises throughput (weights reused) at the cost of per-request latency; **continuous batching** keeps the GPU busy as requests finish.
* **Speculative decoding** (Leviathan et al. 2022; Chen et al. 2023): a small *draft* model proposes $\gamma$ tokens; the target verifies them in one pass. With per-token acceptance probability $\alpha$ the expected tokens per target pass is
$$\mathbb{E}[\text{tokens}]=\frac{1-\alpha^{\gamma+1}}{1-\alpha}$$
(e.g. $\alpha=0.8,\gamma=4\Rightarrow\approx3.4$). The output distribution is *exactly* the target's. Gains depend on draft-target agreement: a distilled small draft of the same family works best.
* **Prompt/prefix caching** saves prefill for shared system prompts (agents).

## 15.4 Serving options to evaluate on your hardware

| Option | Good for | Verify |
| --- | --- | --- |
| Hugging Face `generate` on XPU | development, evals, simplicity | speed is modest; batch via `generate` |
| **llama.cpp** (GGUF) with SYCL or Vulkan backend | low-friction local inference, many quant formats | Battlemage + WSL2 support for your build |
| **OpenVINO GenAI** | Intel-optimised inference | model/format support, WSL2 GPU access |
| **vLLM** | high-throughput serving, LoRA multi-adapter | availability of an XPU build and WSL2 compatibility |
| Ollama/LM Studio-style wrappers | convenience | backends they bundle for Intel GPUs |

Pick one, measure tokens/s and quality on **your** eval set, and keep the exact version in your log.

## 15.5 Model merging (combining fine-tunes)

| Method | What it does | Notes |
| --- | --- | --- |
| **Weight averaging / model soups** (Wortsman et al. 2022) | average weights of fine-tunes from one base | often improves robustness |
| **Task arithmetic** (Ilharco et al. 2022) | add/subtract task vectors $\tau=\theta_{ft}-\theta_{base}$ | simple multi-task composition |
| **TIES** (Yadav et al. 2023) | trim small deltas, resolve sign conflicts, merge | reduces interference |
| **DARE** (Yu et al. 2023) | randomly drop and rescale deltas before merging | sparsifies task vectors |
| **Adapter merging** (PEFT `add_weighted_adapter`) | combine LoRAs | cheap to try; evaluate |

Merging is empirical: there is no guarantee; run your eval and regression suites. Prefer merging adapters fine-tuned from the *same base*.

## 15.6 Packaging and release checklist

1. Merged (or adapter + base reference) weights, tokenizer, **chat template**, `generation_config` (temperature, max tokens, stop tokens).
2. **Model card:** intended use, training data summary and licenses, base model + revision, training method/hyperparameters, evaluation results with CIs, known failure modes, safety limits, hardware/runtime tested, version.
3. **License compliance** of base model, data and outputs.
4. **Regression + safety suites passed;** record the versions.
5. **Serving guardrails:** input length limits, rate limits, schema validation of tool calls, sandboxed tools, logging without sensitive data, rollback plan.
6. **Monitoring:** failure reports, drift in input distribution, periodic re-evaluation on fresh samples.
7. **Versioning:** immutable artifact names with content hashes; keep the previous version deployable.

## Checkpoint

* Merged model matches the adapter outputs (logit diff at bf16 noise level).
* Quantised variant evaluated on your task set with a documented quality/speed/memory trade-off.
* A model card and release checklist exist.
