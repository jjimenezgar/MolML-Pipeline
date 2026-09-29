import pandas as pd
import pytest

from molml.data import load_bace


def test_load_bace_common_moleculenet_schema(tmp_path):
    path = tmp_path / "bace.csv"
    pd.DataFrame({"mol": ["CCO", "CCN"], "Class": [0, 1]}).to_csv(path, index=False)
    df = load_bace(path)
    assert list(df.columns) == ["smiles", "label"]
    assert df["label"].tolist() == [0, 1]


def test_load_bace_rejects_nonbinary_labels(tmp_path):
    path = tmp_path / "bace.csv"
    pd.DataFrame({"mol": ["CCO", "CCN"], "Class": [0, 2]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_bace(path)
