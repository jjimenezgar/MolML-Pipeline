import numpy as np
import pytest

from molml.models import build_model

shap = pytest.importorskip('shap')
from molml.explain import explain_fingerprints


@pytest.mark.parametrize('name', ['logistic_regression', 'random_forest'])
def test_shap_additivity_in_explicit_output_units(name):
    rng = np.random.default_rng(7)
    x = rng.integers(0, 2, size=(20, 16), dtype=np.uint8)
    y = np.array([0, 1] * 10)
    model = build_model(name, 42, {'max_iter': 100} if name == 'logistic_regression'
                        else {'n_estimators': 5, 'n_jobs': 1}).fit(x, y)
    values, base, expected, units, residual, _ = explain_fingerprints(model, x, x[:3], name)
    assert values.shape == (3, 16)
    np.testing.assert_allclose(base + values.sum(axis=1), expected, atol=1e-4)
    assert residual < 1e-4
    assert units == ('log_odds_class_1' if name == 'logistic_regression' else 'probability_class_1')
