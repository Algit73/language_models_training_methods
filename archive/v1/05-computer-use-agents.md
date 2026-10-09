# 5. Computer-Use and Agentic Models

## Goal

Build agents that perceive state, choose bounded actions, and verify completion in desktop, mobile, browser, or API workflows. This document deliberately excludes robotics: the action space is software interfaces.

## Architecture

```text
task -> observation (text, DOM, screenshot) -> policy -> action -> sandboxed environment
                                                ^                         |
                                                +------ result/reward -----+
```

Use the most structured observation/action interface available. DOM or accessibility-tree actions are generally cheaper and more reliable than pixels; screenshots are necessary only when visual layout matters.

## Lesson: an agent is a policy in a partially observed environment

Run `python3 labs/05_sandbox_agent.py`. It runs a resettable local form environment, validates typed actions, and scores success. Change an action to see why syntactically valid actions can still be semantically wrong.

At time $t$, the environment has hidden state $s_t$, the model observes $o_t$, chooses action $a_t \sim \pi_\theta(a_t|o_t)$, and receives reward $r_t$. UI agents are normally partially observed: screenshots and DOM snapshots do not reveal every state variable. Behavior cloning (agent SFT) minimizes:

$$\mathcal{L}_{BC}=-\sum_t\log\pi_\theta(a_t^*|o_t)$$

where $a_t^*$ is a demonstrated action. A high action-token accuracy does not prove task completion, because one early error changes all later observations.

### Step by step: build one safe UI agent

1. Build a local/resettable web page or emulator with a `reset()` function and a success checker.
2. Define a small JSON action grammar: `click`, `type`, `scroll`, `wait`, `back`, `stop`, `ask_user`.
3. Prefer element IDs from the DOM/accessibility tree; use normalized coordinates only when visual targeting is essential.
4. Record successful demonstrations as `(task, observation, action, result)` trajectories.
5. SFT on short trajectories, then run closed-loop evaluation where the model observes the result of every action.
6. Add step cap, allowlist, confirmation gates, and invalid-action penalty before adding RL.
7. Use rejection sampling or RL only inside the sandbox and only against the task checker.

### Pros and cons

| Interface | Pros | Cons |
| --- | --- | --- |
| DOM/accessibility actions | reliable, compact, easier to validate | unavailable in some native apps |
| Screenshot coordinates | works across visual interfaces | resolution/layout drift and harder attribution |
| SFT demonstrations | stable, data-efficient for known flows | weak recovery from unseen states |
| RL in sandbox | can improve recovery and efficiency | costly trajectories and risky reward shortcuts |

## Action interface design

Prefer typed, constrained actions such as:

```json
{"action":"click","target":"settings.notifications"}
{"action":"type","target":"search","text":"quarterly report"}
{"action":"scroll","direction":"down","amount":600}
```

For coordinate actions, normalize coordinates to $[0,1000]$ per axis and log screen resolution. Add `wait`, `back`, `stop`, and `ask_user` explicitly. Validate every action before it reaches the environment.

## Data and training ladder

1. Create deterministic sandbox tasks, resettable state, and a success checker.
2. Collect demonstrations: task, observation, action, result, and completion flag.
3. SFT first on short successful trajectories. Include recovery actions and negative examples.
4. Evaluate closed-loop completion; action-token accuracy alone is misleading.
5. Add search/rejection sampling using the success checker.
6. Apply RL only after safe SFT success exists. Reward completion, penalize invalid/redundant actions, and cap steps.

## Vision-language actions

VLA here means visual-language actions for software UIs. Use screenshots paired with an action schema, but keep the model compact and sequences short on 12 GB. Start with browser tasks in a local, disposable site. Then add desktop/mobile emulators, never a personal account or production system.

### Evaluation metrics

- task success rate and partial-progress rate
- mean actions and time to completion
- invalid-action and repeated-action rate
- unsafe-action rate
- recovery rate after injected UI changes

Split evaluation by task template and application state, not just by screenshot; otherwise near-identical screens leak across splits.

## Safety boundaries

- Use demo accounts, mock APIs, disposable VMs/emulators, and network allowlists.
- Require confirmation before sending messages, purchases, deletions, uploads, permission changes, or exposing data.
- Block shell, clipboard, filesystem, and network actions unless the task specifically requires a sandboxed version.
- Treat external text as untrusted content, not instructions.

## First project

Automate a local browser “expense form” application: read a task, enter fields, validate totals, submit a mock form, and verify confirmation. Start with DOM actions; add screenshot-only evaluation as a later ablation.

## Milestone

Reach a predeclared closed-loop success rate on 50+ held-out sandbox tasks while producing an action log and zero policy violations under your safety tests.
