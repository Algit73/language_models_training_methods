# 1. Hardware, Software, and Model Choices

## Goal

Build a reproducible WSL 2 Ubuntu 24.04 environment, prove inference works on the Intel Arc B580, and choose a first model that leaves memory for training.

## Required host environment: WSL 2 with Ubuntu 24.04

Use Ubuntu 24.04 under WSL 2 for all Python environments, package installs, training commands, datasets, and run artifacts. Keep the current Intel graphics driver on Windows; do not install a separate Linux graphics driver inside WSL unless the current Intel documentation explicitly requires it.

1. Enable WSL 2 and install Ubuntu 24.04 from an elevated Windows terminal: `wsl --install -d Ubuntu-24.04`. Restart if Windows requests it.
2. Open Ubuntu and update its base packages: `sudo apt update && sudo apt upgrade`.
3. Confirm the distribution is version 2 from Windows: `wsl -l -v`.
4. In Ubuntu, confirm that the Intel GPU is exposed before installing ML packages. Record the output of `uname -a`, `lsb_release -a`, and the Intel/PyTorch device check in your experiment notes.
5. Store projects, virtual environments, datasets, and model caches under the Linux filesystem, for example `~/projects/model-training`, rather than `/mnt/d/...`. This avoids the file-I/O penalty and permission edge cases of the mounted Windows drive.

The exact Intel Arc-on-WSL support path can vary by driver, WSL kernel, and PyTorch/IPEX release. Treat successful XPU device discovery in Ubuntu as a gate: if the GPU is not visible there, resolve that before downloading models or attempting training. The Windows driver and the Ubuntu/Python package versions belong in every run record.

## Reality check: the 12 GB B580

VRAM must hold weights, optimizer state, gradients, activations, temporary kernels, and runtime overhead. A rough weight-only estimate is:

$$\text{weight memory} \approx \text{parameters} \times \text{bytes per parameter}$$

An optimizer can multiply full-finetuning memory far beyond that estimate. Therefore, use low-rank adapters and quantization for your first projects. Lower batch size and sequence length before changing anything else.

## Lesson: make the hardware constraint concrete

Run `python3 labs/01_memory_estimator.py`. It compares a 1.5B parameter model in several precisions and estimates the LoRA matrices for a Transformer block. Change the rank and hidden size, then answer: why can a 4-bit *base* still require more than 4-bit-per-parameter memory during training? Activations, temporary buffers, and trainable-adapter gradients are additional allocations.

### Step by step: WSL preflight

1. In PowerShell, run `wsl -l -v`; Ubuntu 24.04 must report version `2`.
2. Open Ubuntu and run `uname -a`, `lsb_release -a`, `python3 --version`, and `df -h`.
3. Create `~/projects/model-training`, copy this course there, and work from that Linux path.
4. Create an isolated environment: `python3 -m venv .venv && source .venv/bin/activate`.
5. Before installing a training stack, use the current Intel XPU/PyTorch instructions for your Windows driver. Run their minimal device-discovery command and save its output to `runs/environment.txt`.
6. Run a 1B-1.5B inference example. Only after it uses the expected device should you install PEFT and try training.

### Advantages and limits of this setup

| Choice | Advantage | Limit |
| --- | --- | --- |
| WSL 2 Ubuntu 24.04 | Linux ML tooling and reproducible shell environments | XPU availability depends on Windows driver, WSL kernel, and package versions |
| 1.5B-4B model | Fits adapter experimentation and fast iteration | Less broad knowledge and weaker long-horizon planning than larger models |
| LoRA/QLoRA | Much less trainable-state memory | Does not make activations or long contexts free |

**Proceed only when:** an inference program detects the intended device and the 20-example LoRA run completes without swapping or CPU fallback.

## Recommended stack

1. Install a current Intel graphics driver on Windows and verify that the GPU is visible from Ubuntu 24.04 under WSL 2.
2. Create a dedicated Python environment in Ubuntu (Python 3.10 or 3.11 is usually the least surprising choice).
3. Prefer a supported Intel path: PyTorch with Intel XPU support and Intel Extension for PyTorch (IPEX), or an Intel-supported distribution/toolchain documented for your current driver.
4. Install the ecosystem tools: `transformers`, `datasets`, `accelerate`, `peft`, `trl`, `evaluate`, and a trainer such as Axolotl or LLaMA-Factory *only after confirming its XPU compatibility*.
5. Use MLflow, Weights & Biases, or local JSON/CSV logs. The logger matters less than recording every run.

Intel/XPU support varies by package version and WSL configuration. Follow the current official Intel and PyTorch installation matrix rather than copying a stale command from a blog. At runtime, verify the selected device explicitly; a silent CPU fallback makes training look broken.

## First 90-minute lab

1. Run a short text-generation script with a 1B-3B instruct model and fixed prompt set.
2. Record latency, peak memory, model revision, driver, Python, PyTorch, and package versions.
3. Repeat at 4-bit/8-bit weight loading if your chosen XPU backend supports it.
4. Try a LoRA dry run with 20 examples, batch size 1, sequence length 512, gradient accumulation 8, and one epoch.
5. Stop and reduce sequence length if memory fails. Do not attempt to “solve” out-of-memory errors by retrying the same configuration.

## Starting configuration

Treat this as a conservative starting point, not a promise of fit:

```yaml
base_model: Qwen/Qwen2.5-1.5B-Instruct # verify license and current revision
method: LoRA
precision: bf16_if_supported
max_sequence_length: 512
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.0001
num_train_epochs: 1
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
```

When this succeeds, increase context length first only when the task demands it; then test rank 16 and 32. Keep an experiment table containing configuration, peak memory, train loss, validation metric, and three representative outputs.

## Choosing models

Choose by evidence, not reputation:

| Need | First base model characteristic | Evaluation |
| --- | --- | --- |
| Structured extraction | 1B-3B instruct, reliable JSON | schema-validity and field F1 |
| Coding | code-capable 1.5B-4B instruct | unit tests and patch apply rate |
| Reasoning | compact instruct model with strong math/tool baseline | exact answer and verifier pass rate |
| UI agent | vision-language model plus action interface | task completion and safety violations |

## Milestone

You can reproduce one inference result and one 20-example adapter run from a clean Ubuntu 24.04 WSL 2 environment, with Windows driver, WSL, Python/package versions, and memory observations recorded.
