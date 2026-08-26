from __future__ import annotations

from pathlib import Path
import sys


PACKAGE = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE.parents[1]
for path in (REPOSITORY_ROOT, PACKAGE):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

# Other experiment suites intentionally use a top-level ``campaign`` module name.
# Combined collection may have cached one already, so evict only foreign aliases
# before this suite imports its sealed package.
for module_name in ("campaign", "worker", "stage_gate", "final_eval", "aggregate", "submit"):
    module = sys.modules.get(module_name)
    module_file = Path(getattr(module, "__file__", "")).resolve() if module else None
    if module_file is not None and not module_file.is_relative_to(PACKAGE):
        del sys.modules[module_name]
