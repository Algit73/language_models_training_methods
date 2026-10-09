# 9. Capstone and Reading List

## Capstone: build a specialist model

Choose one: verified SQL agent, local browser form agent, code patch assistant, grounded meeting-note extractor, or schema-constrained document extractor.

## Twelve-session learning plan

1. Run the memory lab; complete WSL/XPU inference preflight.
2. Run the SFT loss lab; construct and split a 100-example dataset.
3. Train one LoRA adapter and produce a base-vs-adapter evaluation table.
4. Run the verifier lab; build and test a checker for one task.
5. Create verified candidates and compare regular SFT against rejection-sampled SFT.
6. Run the preference lab; create 50 explainable chosen/rejected pairs.
7. Train/evaluate DPO; do not proceed unless it improves held-out task metrics.
8. Run the agent lab and build a resettable environment for your selected specialization.
9. Build a clean evaluation harness and failure taxonomy.
10. Attempt small group-based RL only if you have a deterministic reward and nonzero SFT success.
11. Write data and model cards, reproduce training from an empty environment.
12. Present metrics, 20 failures, safety limits, and the next experiment.

### Deliverables

1. Task specification and safety boundaries.
2. Data card: sources, permissions, annotation guide, splits, and known gaps.
3. Baseline report for prompt-only and base-model performance.
4. SFT adapter and reproducible training configuration.
5. Optional preference/RL experiment justified by a measurable reward.
6. Held-out evaluation report with aggregate metrics and 20 inspected examples.
7. Model card: intended use, unsupported use, limitations, license, hardware, and reproducibility commands.

## Decision guide

| Observation | Next move |
| --- | --- |
| Prompting already meets target | deploy prompt/RAG; do not train |
| Model lacks domain language | CPT on licensed domain text, then SFT |
| Format/behavior is inconsistent | improve SFT data and validators |
| There are clear preferred/rejected outputs | DPO-style preference optimization |
| Success can be automatically checked | rejection sampling, then cautious RL |
| Agent fails in execution | improve environment/tool design and SFT trajectories before RL |

## Papers and primary resources

- [Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)](https://arxiv.org/abs/2106.09685): adapter fine-tuning.
- Dettmers et al., *QLoRA* (2023): quantized adapter training.
- Ouyang et al., *Training language models to follow instructions with human feedback* (2022): the InstructGPT/RLHF pipeline.
- [Rafailov et al., *Direct Preference Optimization* (2023)](https://arxiv.org/abs/2305.18290): preference learning without an explicit reward-model/PPO loop.
- [DeepSeek-AI, *DeepSeekMath* (2024)](https://arxiv.org/abs/2402.03300): group relative policy optimization (GRPO) for verifiable reasoning.
- Yao et al., *ReAct* (2022): interleaving reasoning and tool actions.
- Nakano et al., *WebGPT* (2021) and Zhou et al., *WebArena* (2023): web agents and evaluation.
- Liu et al., *AgentBench* (2023): benchmark suites for language agents.
- Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed.): RL foundations.
- Hugging Face Transformers/PEFT/TRL documentation and current Intel GPU/PyTorch XPU documentation: installation and implementation details.

For the implementation details used in the adapter lesson, see the [Hugging Face PEFT LoRA guide](https://huggingface.co/docs/peft/main/en/conceptual_guides/lora). It documents the trainable rank, target modules, scaling, and adapter merging behavior that the lab reduces to first principles.

The LoRA paper explains the low-rank update assumption; the DPO paper derives its pairwise objective from KL-regularized reward maximization; DeepSeekMath introduces GRPO for memory-conscious group comparison. Read each source alongside its matching lab, then change one quantity and explain the result before using a trainer.


Read papers with a notebook open: identify the environment, data, objective, evaluation, compute budget, and failure cases. Reimplement only the smallest experiment that tests one claim.

## Final standard

You “have your own model” when you can explain the task boundary, reproduce the adapter from a versioned dataset, show a held-out improvement over a baseline, identify failure modes, and deploy it with appropriate action limits. A name on a checkpoint is not the finish line; a measured, maintainable capability is.
