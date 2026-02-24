import sys
from pathlib import Path


# Resolve repo root from this file location to avoid cwd-dependent imports.
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
