"""Make the project modules importable whether the suite is run with
`python -m unittest discover -s tests` or with pytest from any directory."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
