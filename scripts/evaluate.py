"""Replay saved baseline traces and reconcile their accounting without MLX."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plvr3b.baseline import audit_run

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run", type=Path)
args = parser.parse_args()
print(json.dumps(audit_run(args.run), indent=2))
