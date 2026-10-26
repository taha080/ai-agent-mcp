"""Calculator correctness and the allowlist that replaces eval."""

from __future__ import annotations

import ast
import inspect

import pytest

from ai_agent_mcp.tools import calculator
from ai_agent_mcp.tools.calculator import CalculatorError, evaluate


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", "4"),
        ("(2 + 3) * 4", "20"),
        ("2 ** 8", "256"),
        ("7 // 2", "3"),
        ("7 % 4", "3"),
        ("-3 + 10", "7"),
        ("1e2", "100"),
        ("0.1 + 0.2", "0.3"),
        ("sqrt(144)", "12"),
        ("(17 * 23) + sqrt(144)", "403"),
        ("abs(-8)", "8"),
        ("round(3.14159, 2)", "3.14"),
        ("min(3, 1, 2)", "1"),
        ("max(3, 1, 2)", "3"),
        ("pow(2, 5)", "32"),
        ("2 ** -1", "0.5"),
    ],
)
def test_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert evaluate(expression) == expected


def test_rejects_division_by_zero() -> None:
    with pytest.raises(CalculatorError, match="Division by zero"):
        evaluate("1 / 0")


def test_rejects_empty_expression() -> None:
    with pytest.raises(CalculatorError, match="empty"):
        evaluate("   ")


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo pwned')",
        "eval('1')",
        "exec('print(1)')",
        "open('/etc/passwd')",
        "().__class__.__bases__[0].__subclasses__()",
        "(lambda: 1)()",
        "[].__class__",
        "'2' + '2'",
        "2 ^ 3",
        "9 ** 99999",
        "sqrt(-1)",
        "unknown",
    ],
)
def test_rejects_unsafe_or_unsupported_syntax(expression: str) -> None:
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_implementation_does_not_call_eval() -> None:
    tree = ast.parse(inspect.getsource(calculator))
    banned = {"eval", "exec", "compile"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in banned


def test_rejects_non_string() -> None:
    with pytest.raises(CalculatorError, match="string"):
        evaluate(2)  # type: ignore[arg-type]
