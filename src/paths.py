"""Central path resolver.

Every notebook starts with:

    from src.paths import P
    df = pd.read_csv(P.eicu_dir / "patient.csv.gz")

so paths are never hard-coded in notebooks.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

# Resolve project root robustly: <root>/src/paths.py -> root
_ROOT = Path(__file__).resolve().parents[1]
_CONFIG = _ROOT / "configs" / "paths.yml"

with _CONFIG.open("r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)


def _p(key: str) -> Path:
    return Path(_cfg[key])


P = SimpleNamespace(
    root=_ROOT,
    eicu_zip=_p("eicu_zip"),
    mimic_zip=_p("mimic_zip"),
    eicu_dir=_p("eicu_dir"),
    mimic_dir=_p("mimic_dir"),
    interim_dir=_p("interim_dir"),
    processed_dir=_p("processed_dir"),
    figures_dir=_p("figures_dir"),
    tables_dir=_p("tables_dir"),
    cohort=_cfg["cohort"],
)


def ensure_dirs() -> None:
    """Create output directories if they don't exist."""
    for d in (P.interim_dir, P.processed_dir, P.figures_dir, P.tables_dir):
        d.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_dirs()
    for name, value in vars(P).items():
        print(f"{name:18s} {value}")
