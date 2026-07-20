"""
Tiny shared results I/O helper.
================================
Every phase already puts ``phase0/`` on ``sys.path`` (see each run.py and
tests/conftest.py), so importing ``results_io`` from here is the DRY way for all
phases to persist their headline metrics in one consistent place.

Why persist at all?
  * Phase 1 loads Phase 0's baseline instead of hardcoding it.
  * ``scripts/build_results.py`` reads every phase's results.json to regenerate
    the scoreboard tables in ``docs/results.md`` -- so the numbers in the docs can
    never silently drift from what the code actually produced.
  * ``scripts/build_report.py`` renders the same JSON into a flat HTML dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_results(phase_dir: str | Path, payload: dict[str, Any]) -> Path:
    """Write ``payload`` to ``<phase_dir>/results.json`` (pretty-printed)."""
    path = Path(phase_dir) / "results.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_results(phase_dir: str | Path) -> dict[str, Any] | None:
    """Load ``<phase_dir>/results.json``; return None if it doesn't exist."""
    path = Path(phase_dir) / "results.json"
    if path.exists():
        return json.loads(path.read_text())
    return None
