"""Discovery and execution of student scoring modules."""

from __future__ import annotations

import importlib
import importlib.util
import math
import sys
import warnings
from pathlib import Path
from types import ModuleType

from logic.types import Candidate

# Files to skip inside the modules/ directory
_SKIP_FILES = {"__init__.py", "template_module.py"}


def discover_modules(modules_dir: str | Path = "modules") -> list[str]:
    """List available scoring modules in the modules/ directory.

    Ignores __init__.py, template_module.py and files starting with _.
    Returns file stems sorted alphabetically.
    """
    modules_dir = Path(modules_dir)
    if not modules_dir.is_dir():
        return []

    names = []
    for f in sorted(modules_dir.iterdir()):
        if not f.suffix == ".py":
            continue
        if f.name in _SKIP_FILES:
            continue
        if f.name.startswith("_"):
            continue
        names.append(f.stem)
    return names


def run_modules(
    candidates: list[Candidate],
    modules_dir: str | Path = "modules",
) -> list[Candidate]:
    """Run all discovered modules on every candidate.

    Import and runtime errors are caught and reported via warnings
    without crashing the pipeline.
    """
    modules_dir = Path(modules_dir)

    # Add the project root to sys.path if necessary
    project_root = str(modules_dir.resolve().parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    module_names = discover_modules(modules_dir)

    for module_name in module_names:
        module = _import_module_safe(module_name, modules_dir)
        if module is None:
            continue

        for candidate in candidates:
            result = _call_module_safe(module_name, module, candidate)
            if result is not None:
                score_name, score_value = result
                candidate.scores[score_name] = score_value

    return candidates


def _import_module_safe(module_name: str, modules_dir: Path) -> ModuleType | None:
    """Import a module safely. Returns None on error."""
    module_path = modules_dir / f"{module_name}.py"
    try:
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            warnings.warn(f"[{module_name}] Unable to load module spec.")
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        warnings.warn(f"[{module_name}] Import error: {exc}")
        return None


def _call_module_safe(
    module_name: str, module: ModuleType, candidate: Candidate
) -> tuple[str, float] | None:
    """Call get_score safely with triple isolation.

    Returns (score_name, score_value) or None on error.
    """
    # 1. Check that get_score exists
    if not hasattr(module, "get_score"):
        warnings.warn(f"[{module_name}] No get_score() function found.")
        return None

    # 2. Runtime call
    try:
        result = module.get_score(candidate)
    except Exception as exc:
        warnings.warn(
            f"[{module_name}] Runtime error on {candidate.candidate_id}: {exc}"
        )
        return None

    # 3. Return value validation
    if not isinstance(result, tuple) or len(result) != 2:
        warnings.warn(f"[{module_name}] Invalid return (expected 2-tuple): {result}")
        return None

    score_name, score_value = result

    if not isinstance(score_name, str) or not score_name:
        warnings.warn(f"[{module_name}] Invalid score_name: {score_name!r}")
        return None

    if not isinstance(score_value, (int, float)):
        warnings.warn(f"[{module_name}] Non-numeric score_value: {score_value!r}")
        return None

    score_value = float(score_value)
    if math.isnan(score_value) or math.isinf(score_value):
        warnings.warn(f"[{module_name}] NaN/inf score_value: {score_value}")
        return None

    return score_name, score_value
