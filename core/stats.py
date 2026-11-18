"""Small shared statistics helpers (blocking + bootstrap indices)."""
import numpy as np


def blocks(data, size):
    if len(data) % size or len(data) // size < 3:
        raise ValueError('Need at least three complete blocks; no rows may be dropped')
    return data.reshape(len(data) // size, size, *data.shape[1:]).mean(axis=1)


def bootstrap_indices(nblocks, count, seed):
    return np.random.default_rng(seed).integers(0, nblocks, size=(count, nblocks))
