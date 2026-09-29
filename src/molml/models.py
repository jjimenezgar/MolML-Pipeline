"""Classical ML baselines."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_model(name: str, seed: int = 42, parameters: dict | None = None):
    if name == "logistic_regression":
        defaults = {"max_iter": 2000, "class_weight": "balanced"}
        return LogisticRegression(random_state=seed, **(defaults if parameters is None else parameters))
    if name == "random_forest":
        defaults = {"n_estimators": 300, "class_weight": "balanced", "n_jobs": -1}
        return RandomForestClassifier(random_state=seed, **(defaults if parameters is None else parameters))
    raise ValueError(f"Unknown model: {name}")
