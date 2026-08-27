from __future__ import annotations

from pathlib import Path
import sys


PACKAGE = Path(__file__).resolve().parents[1]
value = str(PACKAGE)
while value in sys.path:
    sys.path.remove(value)
sys.path.insert(0, value)
