"""
run_all.py -- run every phase end-to-end on the real KuaiRand-Pure data.
========================================================================
This is the "one obvious way" (Zen) to reproduce the whole repo. It runs each
phase's run.py in order, times it, records pass/fail, and -- once the phases have
written their results.json files -- regenerates the scoreboard (docs/results.md)
and the flat HTML report (docs/report.html).

Usage:
    python run_all.py                # run all phases, then rebuild docs
    python run_all.py --phases 0 1 8 # run a subset (deps permitting)
    python run_all.py --docs-only    # skip phases, just rebuild docs from JSON

Notes:
  * Needs the KuaiRand-Pure CSVs in data/ (see data/README.md). The TEST suite
    does not -- that's `pytest`, which is what CI runs.
  * Phase 1 trains a torch model (~2 min on CPU); the rest are quick.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
# Order matters: phase1 reads phase0's results.json; later phases reuse earlier
# artifacts conceptually. 0 must run before 1 for the baseline comparison.
PHASES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]


def run_phase(n: int) -> tuple[bool, float]:
    """Run phaseN/run.py in its own directory. Returns (ok, seconds)."""
    phase_dir = ROOT / f"phase{n}"
    print(f"\n{'=' * 70}\n  PHASE {n}  ->  {phase_dir.relative_to(ROOT)}/run.py\n{'=' * 70}")
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, "run.py"], cwd=phase_dir)
    elapsed = time.perf_counter() - start
    ok = proc.returncode == 0
    print(f"\n  phase {n}: {'OK' if ok else 'FAILED'} in {elapsed:.1f}s")
    return ok, elapsed


def rebuild_docs() -> None:
    """Regenerate the scoreboard + HTML report from the phase results.json files."""
    for script in ("build_results.py", "build_report.py"):
        path = ROOT / "scripts" / script
        if path.exists():
            print(f"\n[run_all] Rebuilding docs via scripts/{script} ...")
            subprocess.run([sys.executable, str(path)], cwd=ROOT, check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run all phases + rebuild docs.")
    ap.add_argument("--phases", type=int, nargs="*", default=PHASES,
                    help="subset of phase numbers to run (default: all)")
    ap.add_argument("--docs-only", action="store_true",
                    help="skip phases; just rebuild docs from existing results.json")
    args = ap.parse_args()

    if args.docs_only:
        rebuild_docs()
        return 0

    summary: list[tuple[int, bool, float]] = []
    for n in args.phases:
        ok, secs = run_phase(n)
        summary.append((n, ok, secs))

    rebuild_docs()

    print(f"\n{'=' * 70}\n  SUMMARY\n{'=' * 70}")
    print(f"  {'phase':<8}{'status':<10}{'seconds':>10}")
    print(f"  {'-' * 28}")
    for n, ok, secs in summary:
        print(f"  {n:<8}{'OK' if ok else 'FAILED':<10}{secs:>10.1f}")
    total = sum(s for _, _, s in summary)
    failed = [n for n, ok, _ in summary if not ok]
    print(f"  {'-' * 28}\n  total: {total:.1f}s | "
          f"{'all passed' if not failed else 'FAILED: ' + str(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
