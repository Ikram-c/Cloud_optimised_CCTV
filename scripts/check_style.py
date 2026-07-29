#!/usr/bin/env python3
"""Repo style gate: the exact rules this codebase maintains.

Checks every Python file under src, scripts, and tests for lines over
79 columns, tab characters, trailing whitespace, and comments outside
docstrings (shebangs excluded). Library code under src additionally
follows the applicable NASA Power of 10 rules statically: simple
control flow (no recursion, no while loops - every loop is a bounded
for), and no function longer than 60 lines. Exits non-zero with a
per-violation listing, so CI fails loudly and locally reproducibly.

Usage:
    python scripts/check_style.py
"""

import ast
import sys
import tokenize
from pathlib import Path

MAX_COLUMNS = 79
MAX_FUNCTION_LINES = 60
CHECK_DIRS = ("src", "scripts", "tests")
POWER_OF_10_DIRS = ("src",)


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


def power_of_10_violations(path: Path) -> list:
    """Find Power of 10 rule breaches in one file.

    Flags while loops (loops must have fixed for-range bounds),
    direct recursion (simple control flow), and functions longer
    than MAX_FUNCTION_LINES lines.

    Args:
        path (Path): Python file.

    Returns:
        list: (line, reason) pairs.
    """
    found = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            found.append((node.lineno, "while loop (use bounded for)"))
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        length = node.end_lineno - node.lineno + 1
        if length > MAX_FUNCTION_LINES:
            found.append((
                node.lineno,
                f"function {node.name} is {length} lines (> 60)",
            ))
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            callee = inner.func
            if (
                isinstance(callee, ast.Name)
                and callee.id == node.name
            ):
                found.append((
                    inner.lineno,
                    f"recursion in {node.name}",
                ))
    return found


def _check_file(path: Path, rel: Path, po10: bool) -> int:
    """Check one file and print its violations.

    Args:
        path (Path): Python file.
        rel (Path): Path relative to the repo root, for printing.
        po10 (bool): Also apply the Power of 10 checks.

    Returns:
        int: Number of violations found.
    """
    count = 0
    for line, reason in line_violations(path):
        print(f"{rel}:{line}: {reason}")
        count += 1
    for line, text in comment_violations(path):
        print(f"{rel}:{line}: comment outside docstring: {text}")
        count += 1
    if po10:
        for line, reason in power_of_10_violations(path):
            print(f"{rel}:{line}: power-of-10: {reason}")
            count += 1
    return count


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
            po10 = dirname in POWER_OF_10_DIRS
            violations += _check_file(path, rel, po10)
    if violations:
        print(f"{violations} style violation(s)")
        return 1
    print("style clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
