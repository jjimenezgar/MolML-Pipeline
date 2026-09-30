import numpy as np
from fastapi.testclient import TestClient

from molml.api import MAX_BATCH_SIZE, create_app
from molml.artifacts import build_model_bundle
from molml.config import ExperimentConfig
from molml.features import fingerprint_matrix
from molml.models import build_model


def _bundle():
    smiles = ["c1ccccc1", "Cc1ccccc1", "CCO", "CCN", "C1CCCCC1", "C1CCCO1"]
    labels = np.array([0, 1, 0, 1, 1, 0])
    config = ExperimentConfig(
        model="logistic_regression", split="scaffold", seed=42, radius=2, n_bits=256,
        train_fraction=0.8, val_fraction=0.1, test_fraction=0.1,
        model_parameters={"max_iter": 2000, "class_weight": "balanced"}, data_path="unused.csv",
    )
    model = build_model(config.model, config.seed, config.model_parameters)
    model.fit(fingerprint_matrix(smiles, config.radius, config.n_bits), labels)
    return build_model_bundle(model, config, smiles, dataset_sha256="api-test-hash")


def test_health_and_predict_endpoints_reuse_v3_inference():
    with TestClient(create_app(bundle=_bundle())) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_ready"] is True

        response = client.post("/predict", json={"smiles": ["CCO", "not-a-smiles"]})
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert predictions[0]["status"] == "ok"
        assert predictions[0]["prediction"] in (0, 1)
        assert predictions[1]["status"] == "invalid"
        assert predictions[1]["error"]


def test_api_rejects_empty_and_oversized_batches():
    with TestClient(create_app(bundle=_bundle())) as client:
        assert client.post("/predict", json={"smiles": []}).status_code == 422
        oversized = ["CCO"] * (MAX_BATCH_SIZE + 1)
        assert client.post("/predict", json={"smiles": oversized}).status_code == 422


def test_api_reports_not_ready_when_no_model_is_configured():
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"smiles": ["CCO"]}).status_code == 503
