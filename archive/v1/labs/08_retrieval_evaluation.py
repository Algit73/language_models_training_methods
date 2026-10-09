"""Measure whether retrieval returns the known supporting document."""


def tokens(text: str) -> set[str]:
    return set(text.lower().replace("?", "").replace(".", "").split())


def main() -> None:
    documents = {"policy": "Expenses under fifty dollars need a receipt.", "travel": "Travel requires manager approval."}
    query, expected = "When do I need a receipt for expenses?", "policy"
    query_tokens = tokens(query)
    ranked = sorted(documents, key=lambda key: len(query_tokens & tokens(documents[key])), reverse=True)
    print("Ranked document IDs:", ranked)
    print("Recall@1:", float(ranked[0] == expected))


if __name__ == "__main__":
    main()