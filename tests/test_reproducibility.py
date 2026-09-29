import numpy as np
import pandas as pd
import pytest

from molml.config import load_config
from molml.evaluate import classification_metrics
from molml.features import fingerprint_matrix
from molml.models import build_model
from molml.splitting import scaffold_split


def test_config_loading():
    cfg = load_config('configs/rf_scaffold.yaml')
    assert cfg.split == 'scaffold' and cfg.n_bits == 2048
    assert cfg.model_parameters['n_estimators'] == 300


def test_bad_config_rejected(tmp_path):
    path = tmp_path / 'bad.yaml'
    path.write_text('model: random_forest\nsplit: scaffold\n')
    with pytest.raises(ValueError, match='missing'):
        load_config(path)


def test_scaffold_split_deterministic():
    df = pd.DataFrame({'smiles': ['c1ccccc1O', 'c1ccccc1N', 'C1CCCCC1O', 'C1CCCCC1N',
                                  'c1ccncc1O', 'c1ccncc1N', 'C1CCCC1O', 'C1CCCC1N',
                                  'c1ncccc1F', 'c1ncccc1Cl', 'C1CCC1O', 'C1CCC1N'],
                       'label': [0, 1] * 6})
    a = scaffold_split(df, .5, .25, seed=7)
    b = scaffold_split(df, .5, .25, seed=7)
    assert [p.smiles.tolist() for p in a] == [p.smiles.tolist() for p in b]


@pytest.mark.parametrize('model', ['logistic_regression', 'random_forest'])
def test_fixed_seed_predictions(model):
    x = np.tile(np.eye(4, dtype=np.uint8), (5, 1))
    y = np.tile([0, 1, 0, 1], 5)
    first = build_model(model, 42).fit(x, y).predict_proba(x)
    second = build_model(model, 42).fit(x, y).predict_proba(x)
    np.testing.assert_array_equal(first, second)


def test_fingerprint_matrix_dimensions():
    assert fingerprint_matrix(['CCO', 'CCN'], n_bits=512).shape == (2, 512)


def test_metrics_schema():
    result = classification_metrics([0, 1, 0, 1], [0, 1, 0, 0], [.1, .9, .2, .4])
    assert set(result) == {'roc_auc', 'pr_auc', 'f1', 'mcc', 'balanced_accuracy', 'confusion_matrix'}
    assert np.isfinite(list(v for k, v in result.items() if k != 'confusion_matrix')).all()
    assert result['confusion_matrix'] == [[2, 0], [1, 1]]
