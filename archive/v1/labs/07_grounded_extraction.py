"""Check whether extracted claims have exact support in supplied notes."""


def main() -> None:
    notes = "Ava will send the budget on Friday. The team has not selected a venue."
    claims = ["Ava will send the budget on Friday.", "The venue is the downtown office."]
    for claim in claims:
        print(f"Supported={claim in notes}: {claim}")
    print("A production checker should use evidence spans, not only exact matching.")


if __name__ == "__main__":
    main()
