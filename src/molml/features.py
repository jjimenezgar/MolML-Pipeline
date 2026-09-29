"""Molecular validation and fingerprint generation."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


def parse_smiles(smiles: str) -> Chem.Mol:
    """Parse a SMILES string or raise a clear ValueError."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("SMILES must be a non-empty string.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def morgan_fingerprint(smiles: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Return a Morgan bit fingerprint as a NumPy array."""
    mol = parse_smiles(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    array = np.zeros((n_bits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def fingerprint_matrix(smiles: list[str], radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Generate a 2D fingerprint matrix for a list of SMILES."""
    return np.vstack([morgan_fingerprint(s, radius=radius, n_bits=n_bits) for s in smiles])
