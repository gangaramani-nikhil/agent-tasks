"""Real local tools the agent can call.

Each tool is a pydantic-args model plus a plain function. No network access,
no API keys — everything here runs offline.
"""
from __future__ import annotations

import ast
from pathlib import Path

from pydantic import BaseModel, Field

from .runtime import Tool


# --- calculator ------------------------------------------------------------

class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '2 * (3 + 4)'")


_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}
_ALLOWED_UNARYOPS = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported expression element: {ast.dump(node)}")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (no eval())."""
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree)
    return str(result)


# --- word count --------------------------------------------------------------

class WordCountArgs(BaseModel):
    text: str = Field(description="The text to count words/lines/characters in")


def word_count(text: str) -> dict:
    """Count words, lines, and characters in a piece of text."""
    return {
        "words": len(text.split()),
        "lines": text.count("\n") + (1 if text else 0),
        "characters": len(text),
    }


# --- read file -----------------------------------------------------------------

class ReadFileArgs(BaseModel):
    path: str = Field(description="Path to a text file to read")
    max_chars: int = Field(default=4000, description="Truncate the file at this many characters")


def read_file(path: str, max_chars: int = 4000) -> str:
    """Read a local text file (truncated at max_chars).

    TODO: sandbox this to a configured working directory before exposing it
    to a real model.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {path}")
    content = p.read_text(encoding="utf-8", errors="replace")
    if len(content) > max_chars:
        return content[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    return content


def default_tools() -> list[Tool]:
    """The built-in tool set."""
    return [
        Tool(
            name="calculator",
            description="Evaluate a basic arithmetic expression (+-*/ // % ** and parentheses).",
            args_model=CalculatorArgs,
            fn=calculator,
        ),
        Tool(
            name="word_count",
            description="Count words, lines, and characters in a piece of text.",
            args_model=WordCountArgs,
            fn=word_count,
        ),
        Tool(
            name="read_file",
            description="Read a local text file, truncated at max_chars.",
            args_model=ReadFileArgs,
            fn=read_file,
        ),
    ]
