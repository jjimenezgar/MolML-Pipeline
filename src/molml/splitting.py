"""Random and Bemis-Murcko scaffold splitting."""

from __future__ import annotations

from collections import defaultdict
import random
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def random_split(df, train_fraction=0.8, val_fraction=0.1, seed=42):
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


def _scaffold_groups(df):
    groups = defaultdict(list)
    for idx, smiles in enumerate(df["smiles"]):
        groups[bemis_murcko_scaffold(smiles)].append(idx)
    return list(groups.values())


def scaffold_split(df, train_fraction=0.8, val_fraction=0.1, seed=42, max_attempts=200):
    """Seeded scaffold-group split with no scaffold leakage.

    Scaffold groups remain indivisible. Multiple deterministic seeded orderings are
    tried because a naive largest-first allocation can produce a single-class
    validation/test set on BACE. The best valid allocation minimizes deviation
    from requested split sizes and the global positive-class fraction.
    """
    test_fraction = 1.0 - train_fraction - val_fraction
    if min(train_fraction, val_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be positive.")
    if df["label"].nunique() != 2:
        raise ValueError("Scaffold classification split requires two classes.")

    groups = _scaffold_groups(df)
    n = len(df)
    targets = [train_fraction * n, val_fraction * n, test_fraction * n]
    global_rate = float(df["label"].mean())
    best = None

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        ordered = groups.copy()
        rng.shuffle(ordered)
        # Preserve a mild size preference while allowing seeded variation among
        # similarly sized scaffold families.
        ordered.sort(key=lambda g: -(len(g) + rng.random() * 2.0))

        bins = [[], [], []]
        for group in ordered:
            # Assign to the partition furthest below its requested capacity.
            deficits = [targets[i] - len(bins[i]) for i in range(3)]
            destination = max(range(3), key=lambda i: deficits[i])
            bins[destination].extend(group)

        parts = [df.iloc[idx].reset_index(drop=True) for idx in bins]
        if any(part.empty or part["label"].nunique() < 2 for part in parts):
            continue

        size_error = sum(abs(len(parts[i]) - targets[i]) / n for i in range(3))
        balance_error = sum(abs(float(p["label"].mean()) - global_rate) for p in parts)
        score = size_error + 0.25 * balance_error
        if best is None or score < best[0]:
            best = (score, parts)

    if best is None:
        raise ValueError(
            "Could not construct scaffold-disjoint train/validation/test sets "
            "containing both classes. Try different fractions or inspect the data."
        )
    return tuple(best[1])
