"""V1.1: training-only applicability diagnostics and optional MLflow logging."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pandas as pd

from molml.config import load_config
from molml.domain import domain_summary, similarity_domain
from molml.train import fit_experiment


def run_v11(config, output_dir: Path, *, mlflow_uri: str | None = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    model, parts, features, _, _ = fit_experiment(config)
    train_smiles = parts['train']['smiles'].tolist()
    result = {
        'method': 'max_train_tanimoto_morgan_leave_one_out_5th_percentile',
        'config': asdict(config),
        'dataset_sha256': hashlib.sha256(Path(config.data_path).read_bytes()).hexdigest(),
        'notes': 'Similarity coverage is a heuristic, not calibrated uncertainty or evidence of biological activity.',
    }
    for name in ('validation', 'test'):
        part = parts[name]
        domain = similarity_domain(train_smiles, part['smiles'].tolist(),
                                   radius=config.radius, n_bits=config.n_bits)
        result[name] = domain_summary(domain)
        table = pd.DataFrame({
            'partition_row': range(len(part)),
            'label': part['label'].to_numpy(),
            'predicted_probability': model.predict_proba(features(part))[:, 1],
            'nearest_train_tanimoto': domain['nearest_train_similarity'],
            'in_similarity_domain': domain['in_domain'],
            'novel_scaffold': domain['novel_scaffold'],
        })
        table.to_csv(output_dir / f'{name}_domain.csv', index=False)
    (output_dir / 'domain.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if mlflow_uri:
        log_mlflow(result, output_dir, mlflow_uri)
    return result


def log_mlflow(result, output_dir, uri):
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError('Install the optional MLflow extra: pip install -e ".[tracking]"') from exc
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment('MolML-Pipeline-V1.1')
    cfg = result['config']
    with mlflow.start_run(run_name=f"{cfg['model']}_{cfg['split']}_seed{cfg['seed']}"):
        mlflow.log_params({k: str(v) for k, v in cfg.items() if k != 'model_parameters'})
        mlflow.log_params({f'model_{k}': str(v) for k, v in cfg['model_parameters'].items()})
        mlflow.set_tag('dataset_sha256', result['dataset_sha256'])
        mlflow.set_tag('domain_method', result['method'])
        for partition in ('validation', 'test'):
            mlflow.log_metrics({f'{partition}_{key}': value for key, value in result[partition].items()
                                if isinstance(value, (int, float))})
        mlflow.log_artifacts(str(output_dir))


def main():
    parser = argparse.ArgumentParser(description='V1.1 similarity domain and optional experiment tracking')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mlflow-uri', help='Opt-in MLflow tracking URI, e.g. sqlite:///mlflow.db')
    args = parser.parse_args()
    print(json.dumps(run_v11(load_config(args.config), args.output, mlflow_uri=args.mlflow_uri), indent=2))


if __name__ == '__main__':
    main()
