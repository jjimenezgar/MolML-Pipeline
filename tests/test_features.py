import numpy as np
import pytest

from molml.features import morgan_fingerprint, parse_smiles


def test_parse_valid_smiles():
    assert parse_smiles("CCO").GetNumAtoms() == 3


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        parse_smiles("this-is-not-smiles")


def test_morgan_shape_and_type():
    fp = morgan_fingerprint("CCO", n_bits=256)
    assert fp.shape == (256,)
    assert isinstance(fp, np.ndarray)
    assert set(np.unique(fp)).issubset({0, 1})
