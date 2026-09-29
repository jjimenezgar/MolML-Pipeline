"""Random and Bemis-Murcko scaffold splitting."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def random_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
):
    """Stratified 80/10/10-style random split."""
    test_fraction = 1.0 - train_fraction - val_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive.")

    train, temp = train_test_split(
        df, test_size=val_fraction + test_fraction, random_state=seed,
        stratify=df["label"]
    )
    relative_test = test_fraction / (val_fraction + test_fraction)
    val, test = train_test_split(
        temp, test_size=relative_test, random_state=seed, stratify=temp["label"]
    )
    return tuple(x.reset_index(drop=True) for x in (train, val, test))


def scaffold_split(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
):
    """Deterministic scaffold-group split with no scaffold shared across sets."""
    test_fraction = 1.0 - train_fraction - val_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive.")

    groups = defaultdict(list)
    for idx, smiles in enumerate(df["smiles"]):
        groups[bemis_murcko_scaffold(smiles)].append(idx)

    # Largest scaffold families are assigned first; index tie-break is deterministic.
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    n = len(df)
    train_cutoff = train_fraction * n
    val_cutoff = (train_fraction + val_fraction) * n
    train_idx, val_idx, test_idx = [], [], []

    for group in ordered:
        if len(train_idx) + len(group) <= train_cutoff:
            train_idx.extend(group)
        elif len(train_idx) + len(val_idx) + len(group) <= val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)

    if not train_idx or not val_idx or not test_idx:
        raise ValueError("Scaffold split produced an empty partition.")

    return tuple(
        df.iloc[idx].reset_index(drop=True)
        for idx in (train_idx, val_idx, test_idx)
    )
