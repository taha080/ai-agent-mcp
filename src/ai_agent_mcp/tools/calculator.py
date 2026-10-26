"""Arithmetic evaluator with an explicit AST allowlist.

The calculator never calls ``eval``, ``exec``, or ``compile`` on user input.
Only numeric constants, a fixed set of operators, and a fixed set of functions
are accepted. Anything else is rejected before it runs.
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from ai_agent_mcp.tools.errors import ToolFailure

MAX_EXPRESSION_LENGTH = 500
MAX_NODES = 200
MAX_EXPONENT = 1_000
MAX_ABS_INT = 10**18
MAX_ABS_FLOAT = 1e15

_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_FUNCTIONS = ("abs", "round", "sqrt", "min", "max", "pow")


class CalculatorError(ToolFailure):
    """The expression was empty, unsafe, or mathematically invalid."""


def evaluate(expression: str) -> str:
    """Evaluate ``expression`` and return a plain decimal string.

    Args:
        expression: Arithmetic such as ``(17 * 23) + sqrt(144)``.

    Raises:
        CalculatorError: The expression is rejected or cannot be computed.
    """
    if not isinstance(expression, str):
        raise CalculatorError("Expression must be a string.")
    source = expression.strip()
    if not source:
        raise CalculatorError("Expression is empty.")
    if len(source) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError("Expression is too long.")
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid expression: {exc.msg}.") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalculatorError("Expression is too complex.")
    value = _eval(tree.body)
    return _format(value)


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        return _number(node.value)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise CalculatorError("Unary operator is not allowed.")
        return _finite(op(_eval(node.operand)))
    if isinstance(node, ast.BinOp):
        return _binary(node)
    if isinstance(node, ast.Call):
        return _call(node)
    if isinstance(node, ast.Name):
        raise CalculatorError(f"Unknown name {node.id!r}.")
    if isinstance(node, ast.Attribute):
        raise CalculatorError("Attribute access is not allowed.")
    if isinstance(node, ast.Subscript):
        raise CalculatorError("Indexing is not allowed.")
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
        raise CalculatorError("Collections are not allowed.")
    raise CalculatorError(f"Unsupported syntax: {type(node).__name__}.")


def _binary(node: ast.BinOp) -> int | float:
    op = _BINOPS.get(type(node.op))
    if op is None:
        raise CalculatorError("Operator is not allowed.")
    left = _eval(node.left)
    right = _eval(node.right)
    if isinstance(node.op, ast.Pow):
        return _power(left, right)
    if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
        raise CalculatorError("Division by zero.")
    try:
        result = op(left, right)
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    return _finite(result)


def _call(node: ast.Call) -> int | float:
    if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
        allowed = ", ".join(_ALLOWED_FUNCTIONS)
        raise CalculatorError(f"Only these functions are allowed: {allowed}.")
    if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
        raise CalculatorError("Only positional arguments are allowed.")
    args = [_eval(arg) for arg in node.args]
    name = node.func.id
    if name == "abs":
        _expect_arity(name, args, 1, 1)
        return _finite(abs(args[0]))
    if name == "sqrt":
        _expect_arity(name, args, 1, 1)
        if args[0] < 0:
            raise CalculatorError("sqrt domain error.")
        return _finite(math.sqrt(args[0]))
    if name == "round":
        _expect_arity(name, args, 1, 2)
        return _finite(_round(*args))
    if name == "pow":
        _expect_arity(name, args, 2, 2)
        return _power(args[0], args[1])
    _expect_arity(name, args, 1, None)
    function = min if name == "min" else max
    return _finite(function(args))


def _power(base: int | float, exponent: int | float) -> int | float:
    if isinstance(exponent, float) and not exponent.is_integer() and base < 0:
        raise CalculatorError("Negative base with a fractional exponent.")
    if abs(exponent) > MAX_EXPONENT:
        raise CalculatorError("Exponent is too large.")
    try:
        result = base**exponent
    except ZeroDivisionError as exc:
        raise CalculatorError("Division by zero.") from exc
    except OverflowError as exc:
        raise CalculatorError("Result is too large.") from exc
    if isinstance(result, complex):
        raise CalculatorError("Complex results are not supported.")
    return _finite(result)


def _round(value: int | float, digits: int | float | None = None) -> int | float:
    if digits is None:
        return round(value)
    if isinstance(digits, float):
        if not digits.is_integer():
            raise CalculatorError("round digits must be an integer.")
        digits = int(digits)
    if isinstance(digits, bool) or not isinstance(digits, int):
        raise CalculatorError("round digits must be an integer.")
    if abs(digits) > 10:
        raise CalculatorError("round digits are out of range.")
    return round(value, digits)


def _expect_arity(name: str, args: list[int | float], low: int, high: int | None) -> None:
    count = len(args)
    if count < low or (high is not None and count > high):
        raise CalculatorError(f"{name} received the wrong number of arguments.")


def _number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("Only numeric constants are allowed.")
    return _finite(value)


def _finite(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError("Result is not a real number.")
    if isinstance(value, int):
        if abs(value) > MAX_ABS_INT:
            raise CalculatorError("Result is too large.")
        return value
    if not math.isfinite(value) or abs(value) > MAX_ABS_FLOAT:
        raise CalculatorError("Result is too large.")
    return value


def _format(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    rounded = round(value, 10)
    if rounded.is_integer():
        return str(int(rounded))
    return format(rounded, "f").rstrip("0").rstrip(".")
