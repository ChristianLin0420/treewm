import sys
from pathlib import Path


UPSTREAM_RQL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UPSTREAM_RQL_DIR))

