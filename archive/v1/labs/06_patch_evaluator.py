"""Evaluate an isolated patch with a standard-library unit test."""

import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        project = Path(directory)
        (project / "math_utils.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
        (project / "math_utils.py").write_text("def add(left, right):\n    return left + right\n", encoding="utf-8")
        (project / "test_math_utils.py").write_text("import unittest\nfrom math_utils import add\nclass TestAdd(unittest.TestCase):\n    def test_add(self): self.assertEqual(add(17, 25), 42)\n", encoding="utf-8")
        result = subprocess.run([sys.executable, "-m", "unittest", "test_math_utils.py"], cwd=project, capture_output=True, text=True, check=False)
        print("Exit code:", result.returncode)
        print(result.stdout + result.stderr)


if __name__ == "__main__":
    main()
