import pandas as pd

from molml.splitting import bemis_murcko_scaffold, scaffold_split


def _scaffold_sets(parts):
    return [
        {bemis_murcko_scaffold(s) for s in part["smiles"]}
        for part in parts
    ]


def test_scaffold_generation():
    assert bemis_murcko_scaffold("c1ccccc1O") == "c1ccccc1"


def test_scaffolds_do_not_cross_partitions():
    smiles = [
        "c1ccccc1O", "c1ccccc1N", "c1ccccc1Cl",
        "C1CCCCC1O", "C1CCCCC1N", "C1CCCCC1Cl",
        "c1ccncc1O", "c1ccncc1N",
        "C1CCCC1O", "C1CCCC1N",
        "c1ncccc1F", "c1ncccc1Cl",
    ]
    df = pd.DataFrame({"smiles": smiles, "label": [0,1] * 6})
    parts = scaffold_split(df, train_fraction=0.6, val_fraction=0.2, seed=42)
    sets = _scaffold_sets(parts)
    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])


def test_each_scaffold_partition_contains_both_classes():
    smiles = [
        "c1ccccc1O", "c1ccccc1N",
        "C1CCCCC1O", "C1CCCCC1N",
        "c1ccncc1O", "c1ccncc1N",
        "C1CCCC1O", "C1CCCC1N",
        "c1ncccc1F", "c1ncccc1Cl",
        "C1CCC1O", "C1CCC1N",
    ]
    df = pd.DataFrame({"smiles": smiles, "label": [0,1] * 6})
    train, val, test = scaffold_split(
        df, train_fraction=0.5, val_fraction=0.25, seed=7
    )
    assert all(part["label"].nunique() == 2 for part in (train, val, test))
