from collections.abc import Iterable
from typing import TypeVar

import numpy as np

T = TypeVar("T")


def ket(psi: Iterable[T]) -> np.ndarray:
    return np.reshape(np.asarray(psi), (-1, 1))


def base(position: int, dimension: int) -> np.ndarray:
    vector = np.zeros(dimension)
    vector[position] = 1
    return ket(vector)


def bra(psi: Iterable[T]) -> np.ndarray:
    return np.transpose(np.conjugate(ket(psi)))


def braket(psi: Iterable[T], phi: Iterable[T]) -> complex:
    return complex(bra(psi) @ ket(phi))


def ketbra(psi: Iterable[T], phi: Iterable[T]) -> np.ndarray:
    return ket(psi) @ bra(phi)


def proj(psi: Iterable[T]) -> np.ndarray:
    return ket(psi) @ bra(psi)


def res(matrix: np.ndarray) -> np.ndarray:
    """
    vectorization of X row by row. See also: vec(X)
    """
    return ket(np.reshape(np.asarray(matrix), np.size(matrix)))


def vec(matrix: np.ndarray) -> np.ndarray:
    """
    vectorization of X as vectorization column by column.  See also: res(X)
    """
    return ket(np.reshape(np.asarray(matrix).T, np.size(matrix)))


def max_ent(dim: int = 2) -> np.ndarray:
    """
    maximally entangled state.
    """
    return 1 / np.sqrt(dim) * proj(vec(np.identity(dim)))
