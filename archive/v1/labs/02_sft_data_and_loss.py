"""Compute the SFT cross-entropy formula without a machine-learning framework."""

import math


def cross_entropy(probability_of_target: float) -> float:
    return -math.log(probability_of_target)


def main() -> None:
    confident = cross_entropy(0.80)
    uncertain = cross_entropy(0.20)
    print(f"Loss when the target gets 80% probability: {confident:.3f}")
    print(f"Loss when the target gets 20% probability: {uncertain:.3f}")
    print("SFT lowers this loss only on response tokens selected by the mask.")
    example = {"messages": [{"role": "user", "content": "Extract a date."}, {"role": "assistant", "content": '{"date":"2027-05-12"}'}]}
    print("A training record needs both input context and one consistent target:", example)


if __name__ == "__main__":
    main()
