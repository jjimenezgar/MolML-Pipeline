import pandas as pd

from molml.splitting import bemis_murcko_scaffold, scaffold_split


def test_scaffold_generation():
    assert bemis_murcko_scaffold("c1ccccc1O") == "c1ccccc1"


def test_scaffolds_do_not_cross_partitions():
    smiles = [
        "c1ccccc1O", "c1ccccc1N", "c1ccccc1Cl",
        "C1CCCCC1O", "C1CCCCC1N", "C1CCCCC1Cl",
        "c1ccncc1O", "c1ccncc1N",
        "C1CCCC1O", "C1CCCC1N",
    ]
    df = pd.DataFrame({"smiles": smiles, "label": [0,1,0,1,0,1,0,1,0,1]})
    train, val, test = scaffold_split(df, train_fraction=0.6, val_fraction=0.2)

    sets = []
    for part in (train, val, test):
        sets.append({bemis_murcko_scaffold(s) for s in part["smiles"]})

    assert sets[0].isdisjoint(sets[1])
    assert sets[0].isdisjoint(sets[2])
    assert sets[1].isdisjoint(sets[2])
