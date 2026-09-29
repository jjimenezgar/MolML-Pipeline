"""Training-set-only fingerprint similarity diagnostics.

This is a heuristic coverage indicator, not an uncertainty calibration.
"""

from __future__ import annotations

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from molml.features import parse_smiles
from molml.splitting import bemis_murcko_scaffold


def _fingerprints(smiles, radius, n_bits):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return [generator.GetFingerprint(parse_smiles(s)) for s in smiles]


def similarity_domain(train_smiles, query_smiles, *, radius=2, n_bits=2048, quantile=0.05):
    """Nearest-training Tanimoto; threshold from train leave-one-out similarities.

    A query is in-domain if its maximum Tanimoto to training molecules meets
    the fixed lower-tail threshold. Identical entries in training may give 1.0.
    Empty acyclic Murcko scaffolds are treated as one scaffold by the split.
    """
    if not 0 < quantile < 1 or len(train_smiles) < 2:
        raise ValueError("Need at least two training molecules and a quantile in (0, 1).")
    train = _fingerprints(train_smiles, radius, n_bits)
    train_max = np.array([
        max(DataStructs.BulkTanimotoSimilarity(fp, train[:i] + train[i+1:]))
        for i, fp in enumerate(train)
    ])
    threshold = float(np.quantile(train_max, quantile))
    query = _fingerprints(query_smiles, radius, n_bits)
    nearest = np.array([max(DataStructs.BulkTanimotoSimilarity(fp, train)) for fp in query])
    train_scaffolds = {bemis_murcko_scaffold(s) for s in train_smiles}
    novel = [bemis_murcko_scaffold(s) not in train_scaffolds for s in query_smiles]
    return {
        "threshold": threshold,
        "threshold_quantile": quantile,
        "train_leave_one_out_similarity": train_max,
        "nearest_train_similarity": nearest,
        "in_domain": nearest >= threshold,
        "novel_scaffold": np.asarray(novel, dtype=bool),
    }


def domain_summary(domain):
    similarities = domain["nearest_train_similarity"]
    return {
        "threshold": domain["threshold"],
        "threshold_quantile": domain["threshold_quantile"],
        "count": int(len(similarities)),
        "in_domain_count": int(domain["in_domain"].sum()),
        "in_domain_fraction": float(domain["in_domain"].mean()),
        "novel_scaffold_count": int(domain["novel_scaffold"].sum()),
        "nearest_train_similarity_median": float(np.median(similarities)),
    }
