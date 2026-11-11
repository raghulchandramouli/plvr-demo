"""Exhaustively evaluate the Phase 2 oracle grammar without loading a model."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plvr3b.search import run_gate

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--oracle-only", action="store_true", help="Explicitly select the only implemented oracle mode")
args = parser.parse_args()
summary = run_gate(ROOT, args.output)
print(f"Oracle runtime verified. {summary['canonical_program_count']} canonical programs; {summary['useful_workflow_count']} universally correct workflow. Verdict {summary['verdict']}.")
print(f"Evidence saved to {args.output / 'gate_summary.json'}")
