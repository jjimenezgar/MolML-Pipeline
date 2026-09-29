"""V1.2: limited fingerprint-bit attribution diagnostics for fixed V1 models."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from molml.config import load_config
from molml.train import fit_experiment


def explain_fingerprints(model, x_train, x_query, model_name, *, seed=42,
                         background_size=64):
    """Return class-1 SHAP values with checked additivity and explicit units."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError('Install the optional SHAP extra: pip install -e ".[explain]"') from exc
    if x_train.ndim != 2 or x_query.ndim != 2 or x_train.shape[1] != x_query.shape[1]:
        raise ValueError('Train and query fingerprints must have the same bit dimension.')
    if len(x_query) == 0 or len(x_train) == 0:
        raise ValueError('Fingerprints must be nonempty.')
    if model_name == 'logistic_regression':
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(x_train), size=min(background_size, len(x_train)), replace=False)
        background = x_train[indices].astype(float)
        explanation = shap.LinearExplainer(model, shap.maskers.Independent(background))(x_query)
        values = np.asarray(explanation.values)
        base = np.asarray(explanation.base_values)
        expected = model.decision_function(x_query)
        units = 'log_odds_class_1'
    elif model_name == 'random_forest':
        # For sklearn RandomForestClassifier, raw TreeExplainer output is
        # the two-class probability vector; no evaluation data is background.
        explanation = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')(x_query)
        values = np.asarray(explanation.values)[:, :, 1]
        base = np.asarray(explanation.base_values)[:, 1]
        expected = model.predict_proba(x_query)[:, 1]
        units = 'probability_class_1'
    else:
        raise ValueError(f'Unsupported model: {model_name}')
    if values.shape != x_query.shape:
        raise ValueError(f'Unexpected SHAP shape {values.shape}.')
    residual = np.abs(base + values.sum(axis=1) - expected)
    if not np.isfinite(values).all() or not np.isfinite(residual).all() or residual.max() > 1e-4:
        raise ValueError(f'SHAP additivity failed; maximum residual {residual.max()}.')
    return values, base, expected, units, float(residual.max()), shap.__version__


def run_explanation(config, output_dir: Path, *, n_samples=8, top_bits=20):
    if n_samples < 1 or top_bits < 1:
        raise ValueError('Sample and bit counts must be positive.')
    output_dir.mkdir(parents=True, exist_ok=True)
    model, parts, features, _, _ = fit_experiment(config)
    train = features(parts['train'])
    query_frame = parts['test'].iloc[:n_samples]
    query = features(query_frame)
    values, base, output, units, residual, shap_version = explain_fingerprints(
        model, train, query, config.model, seed=config.seed)
    importance = np.mean(np.abs(values), axis=0)
    order = np.argsort(-importance, kind='stable')[:top_bits]
    summary = {
        'config': asdict(config),
        'dataset_sha256': hashlib.sha256(Path(config.data_path).read_bytes()).hexdigest(),
        'shap_version': shap_version,
        'explained_partition': 'test',
        'sample_selection': 'first_n_rows_of_fixed_test_split',
        'sample_count': len(query),
        'linear_background': 'seeded_64_training_rows' if config.model == 'logistic_regression' else None,
        'units': units,
        'max_additivity_residual': residual,
        'top_bits_by_mean_absolute_shap': [
            {'bit_index': int(i), 'mean_absolute_shap': float(importance[i]),
             'training_prevalence': float(train[:, i].mean())} for i in order
        ],
        'limitations': 'Attributions explain model outputs for hashed, collision-prone Morgan bits, not unique molecular substructures or biological mechanisms.',
    }
    details = []
    for row, (b, score, contributions) in enumerate(zip(base, output, values)):
        local_order = np.argsort(-np.abs(contributions), kind='stable')[:min(10, len(contributions))]
        details.append({'partition_row': row, 'true_label': int(query_frame['label'].iloc[row]),
                        'base_value': float(b), 'model_output': float(score),
                        'top_local_bits': [{'bit_index': int(i), 'value': float(contributions[i]),
                                            'fingerprint_bit': int(query[row, i])} for i in local_order]})
    (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (output_dir / 'local_attributions.json').write_text(json.dumps(details, indent=2, allow_nan=False) + '\n')
    fig, ax = plt.subplots(figsize=(8, max(4, len(order) * .27)))
    ax.barh([f'Bit {i}' for i in reversed(order)], importance[order][::-1], color='#3975a8')
    ax.set_xlabel(f'Mean |SHAP value| ({units})')
    ax.set_title(f'{config.model}: {len(query)} fixed test molecules')
    fig.tight_layout()
    fig.savefig(output_dir / 'fingerprint_bits.png', dpi=150)
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description='V1.2 fixed-model fingerprint attribution')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(run_explanation(load_config(args.config), args.output, n_samples=args.samples), indent=2))


if __name__ == '__main__':
    main()
