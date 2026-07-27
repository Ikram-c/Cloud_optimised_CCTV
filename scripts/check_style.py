#!/usr/bin/env python3
"""Repo style gate: the exact rules this codebase maintains.

Checks every Python file under src, scripts, and tests for lines over
79 columns, tab characters, trailing whitespace, and comments outside
docstrings (shebangs excluded). Exits non-zero with a per-violation
listing, so CI fails loudly and locally reproducibly.

Usage:
    python scripts/check_style.py
"""

import sys
import tokenize
from pathlib import Path

MAX_COLUMNS = 79
CHECK_DIRS = ("src", "scripts", "tests")


def comment_violations(path: Path) -> list:
    """Find comment tokens that are not shebang lines.

    Args:
        path (Path): Python file.

    Returns:
        list: (line, text) pairs for offending comments.
    """
    found = []
    with path.open("rb") as handle:
        try:
            for token in tokenize.tokenize(handle.readline):
                if token.type != tokenize.COMMENT:
                    continue
                if token.start[0] == 1 and token.string.startswith("#!"):
                    continue
                found.append((token.start[0], token.string[:40]))
        except tokenize.TokenizeError:
            found.append((0, "unparseable file"))
    return found


def line_violations(path: Path) -> list:
    """Find over-length, tab, and trailing-whitespace lines.

    Args:
        path (Path): Python file.

    Returns:
        list: (line, reason) pairs.
    """
    found = []
    for number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1,
    ):
        if len(line) > MAX_COLUMNS:
            found.append((number, f"line too long ({len(line)} > 79)"))
        if "\t" in line:
            found.append((number, "tab character"))
        if line != line.rstrip():
            found.append((number, "trailing whitespace"))
    return found


def main() -> int:
    """Entry point.

    Returns:
        int: 0 when clean, 1 when violations exist.
    """
    root = Path(__file__).resolve().parent.parent
    violations = 0
    for dirname in CHECK_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(root)
            for line, reason in line_violations(path):
                print(f"{rel}:{line}: {reason}")
                violations += 1
            for line, text in comment_violations(path):
                print(f"{rel}:{line}: comment outside docstring: {text}")
                violations += 1
    if violations:
        print(f"{violations} style violation(s)")
        return 1
    print("style clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
