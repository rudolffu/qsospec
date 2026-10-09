"""Check LaTeX groups that Sphinx builds do not validate."""

import re
from pathlib import Path

import pytest


DOCS = Path(__file__).resolve().parents[1] / "docs"


def _math_expressions(path):
    text = path.read_text()
    if path.suffix == ".md":
        # Code fences contain literal examples rather than rendered equations.
        text = re.sub(r"^```.*?^```[^\n]*", "", text, flags=re.M | re.S)
        yield from re.findall(r"(?<!\\)\$\$(.*?)\$\$", text, flags=re.S)
        text = re.sub(r"(?<!\\)\$\$.*?\$\$", "", text, flags=re.S)
        yield from re.findall(r"(?<![\\$])\$(?!\$)(.*?)(?<!\\)\$", text)
        return

    yield from re.findall(r":math:`([^`]+)`", text)
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.strip() != ".. math::":
            continue
        indent = len(line) - len(line.lstrip())
        expression = []
        for following in lines[index + 1 :]:
            if not following.strip():
                continue
            if len(following) - len(following.lstrip()) <= indent:
                break
            if not following.lstrip().startswith(":"):
                expression.append(following.strip())
        yield "\n".join(expression)


def _check_braces(expression):
    depth = 0
    for token in re.findall(r"\\.|[{}]", expression):
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            assert depth >= 0, f"Unmatched closing brace: {expression}"
    assert depth == 0, f"Unclosed LaTeX group: {expression}"


@pytest.mark.parametrize(
    "path",
    sorted([*DOCS.rglob("*.rst"), *DOCS.rglob("*.md")]),
    ids=lambda path: str(path.relative_to(DOCS)),
)
def test_documentation_math_braces(path):
    for expression in _math_expressions(path):
        _check_braces(expression)


def test_unclosed_unit_group_is_rejected():
    with pytest.raises(AssertionError, match="Unclosed LaTeX group"):
        _check_braces(r"\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}")
    _check_braces(r"\mathrm{erg\,s^{-1}\,cm^{-2}\,\mathring{A}^{-1}}")
    _check_braces(r"\{x\}")
