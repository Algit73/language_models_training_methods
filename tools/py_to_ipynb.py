"""Convert percent-format scripts ('# %%' cells) into .ipynb notebooks (no dependencies).

    python tools/py_to_ipynb.py labs scripts        # converts every *.py that has '# %%' markers

Cell rules:  '# %% [markdown]' starts a markdown cell (leading '# ' is stripped),
             '# %%'            starts a code cell.
The .py file is the single source of truth: edit it, then re-run this converter.
"""
from __future__ import annotations

import json
import pathlib
import sys
import uuid


def parse(text: str) -> list[dict]:
    cells: list[dict] = []
    kind, buf = None, []

    def flush():
        nonlocal kind, buf
        if kind is None:
            return
        while buf and not buf[-1].strip():
            buf.pop()
        while buf and not buf[0].strip():
            buf.pop(0)
        if buf:
            if kind == "markdown":
                buf = [(ln[2:] if ln.startswith("# ") else ln.lstrip("#")) for ln in buf]
            src = [ln + "\n" for ln in buf]
            src[-1] = src[-1].rstrip("\n")
            cell = {"cell_type": kind, "id": uuid.uuid4().hex[:8], "metadata": {}, "source": src}
            if kind == "code":
                cell.update(execution_count=None, outputs=[])
            cells.append(cell)
        kind, buf = None, []

    for line in text.splitlines():
        if line.startswith("# %% [markdown]"):
            flush()
            kind = "markdown"
        elif line.startswith("# %%"):
            flush()
            kind = "code"
        elif kind is not None:
            buf.append(line)
    flush()
    return cells


def convert(path: pathlib.Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "# %%" not in text:
        return False
    nb = {
        "cells": parse(text),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.with_suffix(".ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


if __name__ == "__main__":
    done = 0
    for folder in sys.argv[1:] or ["labs", "scripts"]:
        for py in sorted(pathlib.Path(folder).glob("*.py")):
            if convert(py):
                done += 1
                print("wrote", py.with_suffix(".ipynb"))
    print(f"{done} notebooks written")
