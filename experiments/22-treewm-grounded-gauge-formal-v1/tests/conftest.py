"""Isolate Exp22's package-local generic module names during combined collection."""

from pathlib import Path
import sys


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE.parents[1]
for path in (REPOSITORY_ROOT, PACKAGE):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)
for name in (
    "campaign", "bind_prerequisites", "raw_exp20_recompute", "worker",
    "stage_gate", "final_eval", "aggregate", "submit",
):
    module = sys.modules.get(name)
    module_path = Path(getattr(module, "__file__", "")).resolve() if module else None
    if module_path is not None and not module_path.is_relative_to(PACKAGE):
        del sys.modules[name]
