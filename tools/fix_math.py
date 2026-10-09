"""Normalise Markdown math so GitHub and VS Code's preview both render it.

    python tools/fix_math.py 01-foundations-transformer-math.md 04-sft-and-peft.md ...

Rules (all discovered by testing against GitHub's renderer):
  1. A closing '$' must not be followed by a letter/digit:      $a\\approx3$M   ->  $a\\approx3\\text{M}$
  2. An opening '$' must not be preceded by a letter, '@' or '-': pass@$k$  ->  $\\text{pass@}k$
  3. '$A$-$B$' (a range written as two formulas)                    ->  $A\\text{-}B$
  4. Single-line '$$...$$' is put on three lines ('$$', body, '$$'); GitHub otherwise lets
     markdown emphasis ('_') swallow parts of the formula.
Code fences and `inline code` are never touched.
"""
from __future__ import annotations

import pathlib
import re
import sys

CODE = re.compile(r"`[^`\n]*`")
SPAN = re.compile(r"(?<![\\$])\$(?!\$)((?:[^$\n\\]|\\.)+?)\$(?!\$)")
RUN_BEFORE = re.compile(r"[A-Za-z0-9@\-]+$")
RUN_AFTER = re.compile(r"[A-Za-z0-9]+")


def fix_inline(line: str) -> str:
    stash: list[str] = []

    def hide(m: re.Match) -> str:
        stash.append(m.group(0))
        return f"\x00{len(stash) - 1}\x00"

    line = CODE.sub(hide, line)
    line = re.sub(r"(?<=[^\s$])\$-\$", r"\\text{-}", line)

    out, pos = [], 0
    for m in SPAN.finditer(line):
        if m.start() < pos:
            continue
        before = line[pos:m.start()]
        body = m.group(1)
        lead = RUN_BEFORE.search(before)
        if lead:
            before = before[: lead.start()]
            body = r"\text{" + lead.group(0) + "}" + body
        end = m.end()
        tail = RUN_AFTER.match(line, end)
        if tail:
            body += r"\text{" + tail.group(0) + "}"
            end = tail.end()
        out.append(before + "$" + body + "$")
        pos = end
    out.append(line[pos:])
    line = "".join(out)
    return re.sub(r"\x00(\d+)\x00", lambda m: stash[int(m.group(1))], line)


def fix_text(text: str) -> str:
    result, in_fence = [], False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            result.append(line)
            continue
        if in_fence:
            result.append(line)
            continue
        block = re.match(r"^(\s*)\$\$(.+?)\$\$\s*$", line)
        if block and block.group(2).strip():
            indent = block.group(1)
            result += [f"{indent}$$", f"{indent}{block.group(2).strip()}", f"{indent}$$"]
        else:
            result.append(fix_inline(line))
    return "\n".join(result)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        path = pathlib.Path(name)
        old = path.read_text(encoding="utf-8")
        new = fix_text(old)
        if new != old:
            path.write_text(new, encoding="utf-8", newline="\n")
            print("fixed", name)
