"""Check traceability through the optional real MLflow SQLite backend."""
import json

import pytest

mlflow = pytest.importorskip('mlflow')

from molml.v11 import log_mlflow


def test_tracking_persists_provenance_metrics_and_artifacts(tmp_path):
    uri = f"sqlite:///{tmp_path / 'tracking.db'}"
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    result = {
        'config': {'model': 'random_forest', 'split': 'scaffold', 'seed': 42,
                   'model_parameters': {'n_estimators': 300}},
        'dataset_sha256': 'test-dataset-hash',
        'method': 'training-only-similarity',
        'validation': {'count': 10},
        'test': {'count': 12},
    }
    (artifacts / 'domain.json').write_text(json.dumps(result))
    try:
        log_mlflow(result, artifacts, uri)
        client = mlflow.MlflowClient(tracking_uri=uri)
        experiment = client.get_experiment_by_name('MolML-Pipeline-V1.1')
        runs = client.search_runs([experiment.experiment_id])
        assert len(runs) == 1
        run = runs[0]
        assert run.info.status == 'FINISHED'
        assert run.data.tags['dataset_sha256'] == result['dataset_sha256']
        assert run.data.tags['domain_method'] == result['method']
        assert run.data.params['model_n_estimators'] == '300'
        assert run.data.metrics['test_count'] == 12
        downloaded = client.download_artifacts(run.info.run_id, 'domain.json', str(tmp_path))
        from pathlib import Path
        assert json.loads(Path(downloaded).read_text()) == result
    finally:
        mlflow.set_tracking_uri(None)
