# Small-Model Training Curriculum

This curriculum teaches you to build useful language and multimodal models on one Intel Arc B580 (12 GB VRAM), using WSL 2 with Ubuntu 24.04 as the training environment. It focuses on *post-training*: adapting capable open models to a narrow job, measuring the result, and improving it with supervised learning and reinforcement learning (RL). Training a competitive foundation model from scratch is not realistic on this hardware; creating a strong specialist model is.

## The route

Work in this order. Do not move ahead until the milestone for the current document is met.

1. [Hardware, software, and model choices](01-hardware-and-setup.md) - make a reproducible Intel GPU environment and run an inference baseline.
2. [Data and traditional post-training](02-data-and-supervised-finetuning.md) - curate data, then use continued pretraining, SFT, LoRA, and QLoRA.
3. [Reasoning models](03-reasoning.md) - teach verification, tool use, and concise reasoning traces.
4. [Reinforcement learning methods](04-reinforcement-learning.md) - learn preference optimization, PPO-style RL, GRPO, and reward design.
5. [Computer-use and agentic models](05-computer-use-agents.md) - train models that observe and act in desktop/mobile/browser environments.
6. [Coding models](06-coding-models.md) - specialize a model for repositories, patches, and tests.
7. [Drafting and knowledge-work models](07-drafting-and-knowledge-work.md) - train dependable writing, extraction, and structured-output assistants.
8. [Additional practical specializations](08-more-use-cases.md) - cover RAG, classification, extraction, translation, and multimodal understanding.
9. [Capstone and reading list](09-capstone-and-reading.md) - ship and evaluate your own specialist model.

## How to study this course

Each module follows the same loop: read the mechanism, run its matching `labs/` program in Ubuntu 24.04, change one parameter, predict what should happen, and only then run the real-model version. The toy labs have no ML-framework dependency; they expose the computation hidden by high-level training APIs. They are not substitutes for training a transformer.

```bash
cd ~/projects/model-training
python3 labs/01_memory_estimator.py
```

Do modules 1-4 in order. Modules 5-8 are specializations: select one after module 4. The capstone turns that specialization into a reproducible project.

## What “small” means here

| Tier | Parameters | B580 role | Typical use |
| --- | ---: | --- | --- |
| Tiny | 0.5B-1.5B | Full fine-tuning experiments may fit with short contexts | classifiers, extraction, constrained drafting |
| Small | 1.5B-4B | QLoRA/LoRA sweet spot | coding, structured workflows, narrow agents |
| Medium | 7B-8B | Quantized inference; adapter training only if tooling/memory permit | stronger teacher or deployment baseline |

Start with a 1.5B-4B instruct model. Candidates to benchmark include Qwen2.5/Qwen3 1.5B or 3B, Llama 3.2 1B/3B, SmolLM2 1.7B, and Phi-family small models where their licenses fit your use. For visual UI work, begin with a compact vision-language model only after a text-only agent works. Model availability, licenses, and Intel support change quickly: record exact revisions and licenses in every experiment.

## Non-negotiable experimental loop

1. Define one task and a measurable success condition.
2. Freeze an evaluation set before training.
3. Establish a base-model score.
4. Train one small change at a time.
5. Compare against the baseline, inspect failures, and version the dataset, prompts, code, weights, and metrics.

This avoids the most common failure mode: mistaking a more fluent model for a better model.

## Suggested repository layout

```text
project/
  data/{raw,processed,eval}/
  prompts/
  configs/
  scripts/
  runs/
  model-cards/
  notebooks/
```

Never put secrets, private records, or unredacted credentials into `data/` or training runs. Respect model licenses, website terms, privacy constraints, and the action boundaries of any agent environment.
