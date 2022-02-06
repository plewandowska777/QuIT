from collections.abc import Iterable
from typing import Protocol, TypeVar

import numpy as np
from black import main


class Space(Protocol):

    ...


T = TypeVar("T")


class ComplexSpace:
    def __init__(self, dimension: int, name: str) -> None:
        self.dimension = dimension
        self.name = name

    def base_vector(self, position: int) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=complex)
        vector[position] = 1
        return vector

    def ket(self, psi: Iterable[T]) -> np.ndarray:
        return np.reshape(np.asarray(psi), (-1, 1))

    # def braket(self, psi: , phi) -> complex:


# T = TypeVar("T")
class Ket:
    def __init__(self, psi: Iterable[T], name=None) -> None:
        self.psi = np.reshape(np.asarray(psi), (-1, 1))
        self.name = name


def ket(psi: Iterable[T]) -> np.ndarray:
    return np.reshape(np.asarray(psi), (-1, 1))


def bra(psi: Iterable[T]) -> np.ndarray:
    return np.reshape(np.asarray(psi), (1, -1))


def braket(psi: Iterable[T], phi: Iterable[T]) -> complex:
    return complex(bra(psi) @ ket(phi))


def ketbra(psi: Iterable[T], phi: Iterable[T]) -> np.ndarray:
    return ket(psi) @ bra(phi)


if __name__ == "__main__":
    X = ComplexSpace(2, "X")
    Y = ComplexSpace(2, "Y")
    ro = X.ket([2, 3])
    print(ro)
    psi = ket([1, 2])
    phi = bra([3, 4])
    # print( psi @ phi)
    # print( phi @ psi)
    x = [1, 1]
    y = [2, 4]
    print(ketbra(x, y))
    # print(psi)
    # print(psi)
    # print(np.reshape(psi,(-1,1)) @ np.reshape(psi, (1,-1)))
