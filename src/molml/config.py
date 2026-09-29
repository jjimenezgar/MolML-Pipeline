"""Validated, shared experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    split: str
    seed: int
    radius: int
    n_bits: int
    train_fraction: float
    val_fraction: float
    test_fraction: float
    model_parameters: dict
    data_path: str


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment configuration must be a mapping.")
    required = set(ExperimentConfig.__dataclass_fields__)
    if set(raw) != required:
        raise ValueError(f"Configuration keys: missing {sorted(required-set(raw))}; unknown {sorted(set(raw)-required)}")
    cfg = ExperimentConfig(**raw)
    if cfg.model not in {"logistic_regression", "random_forest"} or cfg.split not in {"random", "scaffold"}:
        raise ValueError("Unknown model or split strategy.")
    if not isinstance(cfg.seed, int) or isinstance(cfg.seed, bool) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    if not isinstance(cfg.radius, int) or cfg.radius < 0 or not isinstance(cfg.n_bits, int) or cfg.n_bits <= 0:
        raise ValueError("Invalid fingerprint settings.")
    fractions = (cfg.train_fraction, cfg.val_fraction, cfg.test_fraction)
    if not all(isinstance(x, (float, int)) and x > 0 for x in fractions) or abs(sum(fractions)-1) > 1e-9:
        raise ValueError("Positive split fractions must sum to one.")
    if not isinstance(cfg.model_parameters, dict):
        raise ValueError("model_parameters must be a mapping.")
    return cfg
