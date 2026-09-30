"""Versioned, trusted model bundles for local inference."""

from __future__ import annotations

from pathlib import Path
import tempfile

import joblib
import numpy as np

from molml.domain import similarity_domain
from molml.features import fingerprint_matrix
from molml.splitting import bemis_murcko_scaffold


BUNDLE_FORMAT = "molml-pipeline-model"
BUNDLE_VERSION = 1


def build_model_bundle(model, config, training_smiles, *, dataset_sha256=None):
    """Package a fitted model and training-only references needed at inference."""
    smiles = [str(value) for value in training_smiles]
    if len(smiles) < 2:
        raise ValueError("At least two training molecules are required for a model bundle.")
    radius, n_bits = config.radius, config.n_bits
    domain = similarity_domain(smiles, smiles, radius=radius, n_bits=n_bits)
    return {
        "format": BUNDLE_FORMAT,
        "format_version": BUNDLE_VERSION,
        "model": model,
        "model_name": config.model,
        "classes": [int(value) for value in model.classes_],
        "fingerprint": {"kind": "morgan", "radius": radius, "n_bits": n_bits},
        "decision_threshold": 0.5,
        "similarity_domain": {
            "threshold": float(domain["threshold"]),
            "threshold_quantile": float(domain["threshold_quantile"]),
        },
        "training_fingerprints": fingerprint_matrix(smiles, radius=radius, n_bits=n_bits),
        "training_scaffolds": sorted({bemis_murcko_scaffold(s) for s in smiles}),
        "provenance": {
            "dataset_sha256": dataset_sha256,
            "split": config.split,
            "seed": config.seed,
            "training_molecule_count": len(smiles),
        },
    }


def save_model_bundle(bundle, path: str | Path) -> Path:
    """Write a model bundle atomically. Only load bundles from trusted sources."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
        joblib.dump(bundle, temporary)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_model_bundle(path: str | Path) -> dict:
    """Load a locally generated bundle and validate its public schema.

    Joblib uses pickle internally. Loading a bundle from an untrusted source can
    execute code; use only model files you created or otherwise trust.
    """
    return _validate_bundle(joblib.load(path))


def _validate_bundle(bundle: dict) -> dict:
    required = {"format", "format_version", "model", "classes", "fingerprint",
                "decision_threshold", "similarity_domain", "training_fingerprints",
                "training_scaffolds", "provenance"}
    if not isinstance(bundle, dict) or not required.issubset(bundle):
        raise ValueError("Invalid MolML model bundle: required metadata is missing.")
    if bundle["format"] != BUNDLE_FORMAT or bundle["format_version"] != BUNDLE_VERSION:
        raise ValueError("Unsupported MolML model bundle format or version.")
    fingerprint = bundle["fingerprint"]
    if fingerprint.get("kind") != "morgan" or int(fingerprint.get("n_bits", 0)) <= 0:
        raise ValueError("Invalid fingerprint settings in model bundle.")
    reference = np.asarray(bundle["training_fingerprints"])
    if reference.ndim != 2 or reference.shape[1] != int(fingerprint["n_bits"]) or len(reference) < 2:
        raise ValueError("Invalid training fingerprint reference in model bundle.")
    if not hasattr(bundle["model"], "predict_proba") or not {0, 1}.issubset(bundle["classes"]):
        raise ValueError("Model bundle must support probabilities for classes 0 and 1.")
    return bundle
