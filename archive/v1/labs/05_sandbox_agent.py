"""A resettable, side-effect-free environment for typed UI actions."""


class ExpenseForm:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.fields = {"merchant": "", "amount": ""}
        self.submitted = False

    def act(self, action: dict[str, str]) -> str:
        if action.get("action") == "type" and action.get("target") in self.fields:
            self.fields[action["target"]] = action.get("text", "")
            return "typed"
        if action.get("action") == "submit" and self.fields == {"merchant": "Cafe", "amount": "12"}:
            self.submitted = True
            return "submitted"
        return "invalid_or_incomplete"


def main() -> None:
    environment = ExpenseForm()
    for action in ({"action": "type", "target": "merchant", "text": "Cafe"}, {"action": "type", "target": "amount", "text": "12"}, {"action": "submit"}):
        print(action, "=>", environment.act(action))
    print("Task success:", environment.submitted)


if __name__ == "__main__":
    main()
