"""
Tool definitions and execution for StudyMate.

TOOL_SCHEMAS uses OpenAI function-calling format and is passed directly to
LLMClient.chat().  run_tool() is the single dispatch point — main.py calls
it whenever the model emits a tool_use block.

retrieve_from_notes is intentionally absent here.  It is registered
dynamically by main.py only when a RAG index has been built, so the model
doesn't see it as an option until notes have actually been indexed.
"""

from __future__ import annotations

import ast
import math
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_current_datetime",
            "description": (
                "Returns the current local date and time as an ISO-8601 string. "
                "Use this whenever the student asks about deadlines, how many days "
                "until an exam, or any question that requires knowing today's date."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": (
                "Evaluates a mathematical expression safely and returns the result "
                "as a string. Always use this tool for any arithmetic — never compute "
                "numbers mentally. "
                "Supported operators: +  -  *  /  //  %  ** . "
                "Supported functions: sqrt, log, log2, log10, exp, "
                "sin, cos, tan, asin, acos, atan, degrees, radians, "
                "floor, ceil, abs, round. "
                "Supported constants: pi, e. "
                "Example expressions: '2**10', 'sqrt(144)', 'log2(1024)', "
                "'sin(pi/2)', 'round(3.14159, 2)'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A mathematical expression to evaluate.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Searches the web using DuckDuckGo and returns the top results. "
                "Use for recent events, unfamiliar libraries, documentation, or "
                "anything where training data may be outdated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (1–8, default 5).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "unit_converter",
            "description": (
                "Converts a numeric value between units. "
                "Temperature: C, F, K. "
                "Length: m, km, cm, mm, mi, ft, in. "
                "Mass: kg, g, lb, oz. "
                "Time: s, min, h, d."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "The numeric value to convert.",
                    },
                    "from_unit": {
                        "type": "string",
                        "description": "The unit to convert from.",
                    },
                    "to_unit": {
                        "type": "string",
                        "description": "The unit to convert to.",
                    },
                },
                "required": ["value", "from_unit", "to_unit"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Calculator — AST-based safe evaluator
# ---------------------------------------------------------------------------
# We use Python's ast module to parse the expression into a syntax tree and
# then walk only the node types we explicitly allow.  The alternative —
# eval() — executes arbitrary Python code: a student could pass
# "__import__('os').system('rm -rf /')" and it would run.  The AST approach
# is safe by design: any node type not in our whitelist raises ValueError
# before any computation occurs.

_ALLOWED_FUNCTIONS: dict[str, Any] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "degrees": math.degrees,
    "radians": math.radians,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
}

_ALLOWED_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a ** b,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}


class _SafeCalc(ast.NodeVisitor):
    """Walks an AST and computes the result of an arithmetic expression.

    Only whitelisted node types are permitted.  Any other node type triggers
    generic_visit, which raises ValueError and aborts evaluation immediately.
    """

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Non-numeric literal: {node.value!r}")
        return node.value

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Operator not allowed: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _ALLOWED_BINOPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unary operator not allowed: {op_type.__name__}")
        operand = self.visit(node.operand)
        return _ALLOWED_UNARYOPS[op_type](operand)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id not in _ALLOWED_CONSTANTS:
            raise ValueError(f"Unknown name: {node.id!r} (only 'pi' and 'e' are allowed)")
        return _ALLOWED_CONSTANTS[node.id]

    def visit_Call(self, node: ast.Call) -> Any:
        # Only simple bare-name function calls are allowed (no attribute calls
        # like math.sqrt or chained methods like x.strip()).
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only named functions are allowed (e.g. sqrt(x), not obj.method())")
        name = node.func.id
        if name not in _ALLOWED_FUNCTIONS:
            raise ValueError(
                f"Function not allowed: {name!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_FUNCTIONS))}"
            )
        args = [self.visit(arg) for arg in node.args]
        # keyword arguments are not supported — keep the surface area minimal
        if node.keywords:
            raise ValueError("Keyword arguments are not supported in calculator expressions.")
        return _ALLOWED_FUNCTIONS[name](*args)

    def generic_visit(self, node: ast.AST) -> Any:
        raise ValueError(f"Expression contains a disallowed construct: {type(node).__name__}")


def calculate(expression: str) -> str:
    """Parse and evaluate *expression* safely. Returns the result as a string."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        return f"Syntax error in expression: {exc}"
    try:
        result = _SafeCalc().visit(tree)
    except ZeroDivisionError:
        return "Error: division by zero."
    except ValueError as exc:
        return f"Error: {exc}"
    # Format: strip trailing .0 for whole-number floats to keep output clean
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)


# ---------------------------------------------------------------------------
# Unit converter
# ---------------------------------------------------------------------------

# Linear units: multiply by this factor to get the base unit.
# Length base = metres, mass base = kilograms, time base = seconds.
_TO_BASE: dict[str, float] = {
    # Length
    "m": 1.0,
    "km": 1000.0,
    "cm": 0.01,
    "mm": 0.001,
    "mi": 1609.344,
    "ft": 0.3048,
    "in": 0.0254,
    # Mass
    "kg": 1.0,
    "g": 0.001,
    "lb": 0.45359237,
    "oz": 0.028349523125,
    # Time
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

# Temperature conversions can't be linear (they have offsets), so we store
# two-step lambdas: first convert to Celsius, then from Celsius to target.
_TO_CELSIUS: dict[str, Any] = {
    "C": lambda v: v,
    "F": lambda v: (v - 32) * 5 / 9,
    "K": lambda v: v - 273.15,
}

_FROM_CELSIUS: dict[str, Any] = {
    "C": lambda v: v,
    "F": lambda v: v * 9 / 5 + 32,
    "K": lambda v: v + 273.15,
}

_TEMP_UNITS = frozenset(_TO_CELSIUS)


def convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert *value* from *from_unit* to *to_unit*. Returns a result string."""
    f = from_unit.strip()
    t = to_unit.strip()

    if f == t:
        return f"{value} {t}"

    # Temperature path (non-linear — must use the dedicated lambdas)
    if f in _TEMP_UNITS or t in _TEMP_UNITS:
        if f not in _TEMP_UNITS or t not in _TEMP_UNITS:
            return f"Error: cannot mix temperature unit '{f if f not in _TEMP_UNITS else t}' with a non-temperature unit."
        celsius = _TO_CELSIUS[f](value)
        result = _FROM_CELSIUS[t](celsius)
        return f"{value} {f} = {round(result, 6)} {t}"

    # Linear path: convert to base unit, then to target
    if f not in _TO_BASE:
        return f"Error: unknown unit '{f}'."
    if t not in _TO_BASE:
        return f"Error: unknown unit '{t}'."

    # Guard against mixing unit families (metres and kilograms share the same
    # base-factor table but are physically incompatible).
    _length = {"m", "km", "cm", "mm", "mi", "ft", "in"}
    _mass   = {"kg", "g", "lb", "oz"}
    _time   = {"s", "min", "h", "d"}

    def _family(u: str) -> str:
        if u in _length: return "length"
        if u in _mass:   return "mass"
        if u in _time:   return "time"
        return "unknown"

    fam_f, fam_t = _family(f), _family(t)
    if fam_f != fam_t:
        return f"Error: cannot convert between {fam_f} ({f}) and {fam_t} ({t})."

    result = value * _TO_BASE[f] / _TO_BASE[t]
    return f"{value} {f} = {round(result, 8)} {t}"


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------

def web_search(query: str, max_results: int = 5) -> str:
    """Search DuckDuckGo and return a formatted numbered list of results."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        return "Error: duckduckgo-search package is not installed. Run: pip install duckduckgo-search"

    # Cap at 8 to avoid overwhelming the context window with search noise.
    n = max(1, min(max_results, 8))

    try:
        with DDGS() as ddgs:
            hits = list(ddgs.text(query, max_results=n))
    except Exception as exc:
        return f"Search failed: {exc}"

    if not hits:
        return "No results found."

    lines: list[str] = []
    for i, hit in enumerate(hits, 1):
        title   = hit.get("title", "(no title)")
        url     = hit.get("href", "")
        snippet = hit.get("body", "")
        lines.append(f"{i}. {title}\n   {url}\n   {snippet}")

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Datetime
# ---------------------------------------------------------------------------

def get_current_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_tool(name: str, arguments: dict) -> str:
    """Route a tool call to the correct implementation.

    Returns a string result in all cases — errors are returned as strings
    so the model can read them and adjust its response rather than crashing.
    """
    if name == "get_current_datetime":
        return get_current_datetime()

    if name == "calculator":
        expr = arguments.get("expression", "")
        if not expr:
            return "Error: 'expression' argument is required."
        return calculate(expr)

    if name == "web_search":
        query = arguments.get("query", "")
        if not query:
            return "Error: 'query' argument is required."
        max_results = int(arguments.get("max_results", 5))
        return web_search(query, max_results)

    if name == "unit_converter":
        try:
            value     = float(arguments["value"])
            from_unit = str(arguments["from_unit"])
            to_unit   = str(arguments["to_unit"])
        except (KeyError, TypeError, ValueError) as exc:
            return f"Error: invalid arguments for unit_converter: {exc}"
        return convert(value, from_unit, to_unit)

    return f"Error: unknown tool '{name}'."
