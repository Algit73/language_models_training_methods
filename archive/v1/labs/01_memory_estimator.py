"""Estimate why adapter tuning fits where full fine-tuning does not."""


def gibibytes(byte_count: int) -> float:
    return byte_count / (1024**3)


def main() -> None:
    parameters = 1_500_000_000
    for label, bytes_per_parameter in (("bf16 weights", 2), ("8-bit weights", 1), ("4-bit weights", 0.5)):
        print(f"{label:16}: {gibibytes(int(parameters * bytes_per_parameter)):.2f} GiB")

    hidden_size, rank, projections = 2048, 8, 4
    lora_parameters = projections * rank * (hidden_size + hidden_size)
    print(f"LoRA parameters/block: {lora_parameters:,}")
    print("Increase rank from 8 to 32 and predict the change before running again.")


if __name__ == "__main__":
    main()
