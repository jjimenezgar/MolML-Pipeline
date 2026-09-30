import numpy as np
import pytest

from molml.artifacts import build_model_bundle, load_model_bundle, save_model_bundle
from molml.config import ExperimentConfig
from molml.features import fingerprint_matrix
from molml.models import build_model
from molml.predict import predict_smiles


def _fitted_bundle():
    smiles = ["c1ccccc1", "Cc1ccccc1", "CCO", "CCN", "C1CCCCC1", "C1CCCO1"]
    labels = np.array([0, 1, 0, 1, 1, 0])
    config = ExperimentConfig(
        model="logistic_regression", split="scaffold", seed=42, radius=2, n_bits=256,
        train_fraction=0.8, val_fraction=0.1, test_fraction=0.1,
        model_parameters={"max_iter": 2000, "class_weight": "balanced"}, data_path="unused.csv",
    )
    model = build_model(config.model, config.seed, config.model_parameters)
    model.fit(fingerprint_matrix(smiles, config.radius, config.n_bits), labels)
    return build_model_bundle(model, config, smiles, dataset_sha256="example-hash")


def test_saved_bundle_roundtrip_and_batch_predictions(tmp_path):
    path = tmp_path / "model.joblib"
    save_model_bundle(_fitted_bundle(), path)
    bundle = load_model_bundle(path)
    predictions = predict_smiles(bundle, ["CCO", "not-a-smiles", "c1ccccc1"])

    assert [row["status"] for row in predictions] == ["ok", "invalid", "ok"]
    assert predictions[1]["error"].startswith("Invalid SMILES")
    assert predictions[1]["probability_class_1"] == ""
    assert 0 <= predictions[0]["probability_class_1"] <= 1
    assert 0 <= predictions[0]["nearest_train_similarity"] <= 1
    assert isinstance(predictions[0]["in_similarity_domain"], bool)
    assert isinstance(predictions[0]["novel_scaffold"], bool)


def test_bundle_schema_rejects_invalid_fingerprint_reference():
    bundle = _fitted_bundle()
    bundle["training_fingerprints"] = np.zeros((3, 12), dtype=np.uint8)
    from molml.artifacts import BUNDLE_FORMAT, BUNDLE_VERSION
    bundle["format"] = BUNDLE_FORMAT
    bundle["format_version"] = BUNDLE_VERSION

    # Exercise validation without loading pickle data from an external source.
    from molml.artifacts import _validate_bundle
    with pytest.raises(ValueError, match="fingerprint reference"):
        _validate_bundle(bundle)
