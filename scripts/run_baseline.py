"""Run the pinned local baseline or validate the smoke oracles."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plvr3b.baseline import run_baseline

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True, help="New run directory; existing directories are never overwritten")
parser.add_argument("--limit", type=int, default=30, help="Number of smoke examples, default 30")
parser.add_argument("--oracle-only", action="store_true", help="Validate deterministic oracles without loading MLX")
args = parser.parse_args()
result = run_baseline(ROOT, args.output, args.limit, args.oracle_only)
print(f"Run {result['status']}. Summary saved to {args.output / 'summary.json'}")
sys.exit(0 if result["status"] == "complete" else 1)
