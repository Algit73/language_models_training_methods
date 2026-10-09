"""Demonstrate rejection sampling using a deterministic arithmetic verifier."""

from dataclasses import dataclass


@dataclass
class Candidate:
    reasoning: str
    answer: int


def verify(question: str, candidate: Candidate) -> bool:
    return question == "17 + 25" and candidate.answer == 42


def main() -> None:
    question = "17 + 25"
    candidates = [Candidate("17 + 25 = 41", 41), Candidate("17 + 25 = 42", 42), Candidate("I guess 43", 43)]
    accepted = [candidate for candidate in candidates if verify(question, candidate)]
    print("Generated:", len(candidates), "Accepted:", len(accepted))
    for candidate in accepted:
        print("Keep for verified SFT:", candidate)


if __name__ == "__main__":
    main()
