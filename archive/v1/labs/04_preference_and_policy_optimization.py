"""Calculate a toy DPO loss and group-relative advantages."""

import math
from statistics import mean, pstdev


def dpo_loss(policy_win: float, reference_win: float, policy_lose: float, reference_lose: float, beta: float = 0.1) -> float:
    margin = beta * (math.log(policy_win / reference_win) - math.log(policy_lose / reference_lose))
    return -math.log(1 / (1 + math.exp(-margin)))


def main() -> None:
    print(f"Toy DPO loss: {dpo_loss(0.50, 0.25, 0.10, 0.25):.4f}")
    rewards = [0.0, 1.0, 1.0, 0.0]
    scale = pstdev(rewards) or 1.0
    advantages = [(reward - mean(rewards)) / scale for reward in rewards]
    print("Group rewards:", rewards)
    print("Normalized advantages:", [round(value, 2) for value in advantages])


if __name__ == "__main__":
    main()
