"""Loading and validation for the MoleculeNet BACE benchmark."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

SMILES_CANDIDATES = ("mol", "smiles", "SMILES")
LABEL_CANDIDATES = ("Class", "class", "label", "target")


def _find_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def load_bace(path: str | Path) -> pd.DataFrame:
    """Load BACE and normalize the required columns to smiles/label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"BACE dataset not found at {path}. See README.md for data setup."
        )

    df = pd.read_csv(path)
    smiles_col = _find_column(df.columns, SMILES_CANDIDATES)
    label_col = _find_column(df.columns, LABEL_CANDIDATES)
    if smiles_col is None or label_col is None:
        raise ValueError(
            "Could not identify BACE SMILES/label columns. "
            f"Available columns: {list(df.columns)}"
        )

    out = df[[smiles_col, label_col]].rename(
        columns={smiles_col: "smiles", label_col: "label"}
    )
    out = out.dropna().drop_duplicates(subset=["smiles"]).reset_index(drop=True)
    out["label"] = pd.to_numeric(out["label"], errors="raise").astype(int)

    labels = set(out["label"].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"BACE labels must be binary 0/1; found {sorted(labels)}.")
    return out
