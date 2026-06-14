from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_archetypes_csv(path: str | Path) -> pd.DataFrame:
    """Load SynDX archetypes CSV (synthetic; no patient data)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Archetypes file not found: {path}")
    return pd.read_csv(path)
