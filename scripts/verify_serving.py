"""Audit M12 function size, annotations and prohibited typing escapes."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def audit_serving_structure(root: Path) -> tuple[str, ...]:
    """Audit M12 modules. Args: root. Returns: violations. Raises: OSError/SyntaxError."""
    directories = (root / "api", root / "src/tr_ru_forecast/serving", root / "tests/api")
    violations = []
    for directory in directories:
        for path in sorted(directory.glob("*.py")):
            contents = path.read_text()
            if "type: ignore" in contents:
                violations.append(f"{path.name}: ignored type error")
            for node in ast.walk(ast.parse(contents)):
                if isinstance(node, ast.Name) and node.id == "Any":
                    violations.append(f"{path.name}: Any")
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    violations.extend(_audit_function(path, node))
    return tuple(violations)


def _audit_function(path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, ...]:
    violations = []
    if node.end_lineno is None or node.end_lineno - node.lineno + 1 > 30:
        violations.append(f"{path.name}:{node.name}: exceeds 30 lines")
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    if node.args.vararg:
        arguments.append(node.args.vararg)
    if node.args.kwarg:
        arguments.append(node.args.kwarg)
    if node.returns is None or any(argument.annotation is None for argument in arguments if argument.arg not in {"self", "cls"}):
        violations.append(f"{path.name}:{node.name}: missing annotation")
    if not ast.get_docstring(node) and not node.name.startswith("_"):
        violations.append(f"{path.name}:{node.name}: missing public docstring")
    return tuple(violations)


def main() -> int:
    """Run CLI audit. Args: none. Returns: exit status. Raises: filesystem/syntax errors."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    violations = audit_serving_structure(Path(__file__).resolve().parent.parent)
    for violation in violations:
        logger.error("%s", violation)
    if violations:
        return 1
    logger.info("M12 structure audit passed: <=30 lines, annotated, no Any/type-ignore")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
