from collections.abc import Iterable

import numpy as np
from scipy import linalg


def ket(psi: Iterable[complex]) -> np.ndarray:
    """Create a ket |psi> as a vertical vector.

    :param ket: Sequence of complex numbers.
    :return: A vertical vector as numpy array.
    """
    return np.reshape(np.asarray(psi), (-1, 1))


def basis(dim: int) -> Iterable[np.ndarray]:
    """Create computational basis of dimension dim.

    :param dim: Dimension of vector space.
    :return: Set of {|i>}i where i={0,...,d-1}.
    """
    vec = [np.zeros(dim) for _ in range(dim)]
    for i in range(dim):
        vec[i][i] = 1
    return vec


def bra(psi: Iterable[complex]) -> np.ndarray:
    """Create bra <psi| as a horizontal vector.

    :param psi: Sequence of complex numbers.
    :return: A horizontal vector as numpy array satisfying <psi| = |psi>^+
    """
    return np.transpose(np.conjugate(ket(psi)))


def braket(psi: Iterable[complex], phi: Iterable[complex]) -> complex:
    """Create braket as an inner product between vectors psi and phi.

    :param psi: Sequence of complex numbers.
    :param phi: Sequence of complex numbers.
    :return: Inner product between vectors psi and phi.
    """
    return complex(bra(psi) @ ket(phi))


def ketbra(psi: Iterable[complex], phi: Iterable[complex]) -> np.ndarray:
    """Create a matrix from multiplication vectors psi and phi.

    :param psi: Sequence of complex numbers.
    :param phi: Sequence of complex numbers.
    :return: A matrix |psi><phi|
    """
    return ket(psi) @ bra(phi)


def proj(psi: Iterable[complex]) -> np.ndarray:
    """Create a projector from an unit vector |psi>.

    :param psi: Sequence of complex numbers.
    :return: A matrix |psi><psi|.
    """
    return ket(psi / linalg.norm(psi)) @ bra(psi / linalg.norm(psi))


def dagger(matrix: np.ndarray) -> np.ndarray:
    """Create conjugate transpose of a given matrix.

    :param matrix: A matrix given by numpy array.
    :return: An operator dagger as concatanation conjugate and transpose.
    """
    return np.transpose(np.conjugate(matrix))


def res(matrix: np.ndarray) -> np.ndarray:
    """Reshaping maps matrix into a vector row by row. res(A) is equivalent to vec(A.T).

    :param matrix: A matrix given by numpy array.
    :return: A vertical vector creating by matrix reshaping.
    """
    return ket(np.reshape(np.asarray(matrix), np.size(matrix)))


def unres(psi: Iterable[complex], dims: tuple[int, int]) -> np.ndarray:
    """Un-reshaping of the vector into the matrix of dimension dims.

    :param psi: A matrix given by numpy array.
    :param dims: A tuple of dimensions.
    :return: A vertical vector creating by matrix vectorization
    """
    return np.asarray(np.reshape(np.asarray(psi), dims))


def vec(matrix: np.ndarray) -> np.ndarray:
    """Reshaping maps matrix into a vector column by column. vec(A) is equivalent to res(A.T).

    :param matrix: A matrix given by numpy array.
    :return: A matrix of dimension dims.
    """
    return ket(np.reshape(np.asarray(matrix).T, np.size(matrix)))


def unvec(psi: Iterable[complex], dims: tuple[int, int]) -> np.ndarray:
    """Re-vectorization of the vector into the matrix of dimension dims.

    :param psi: A matrix given by numpy array.
    :param dims: A tuple of dimensions
    :return: A matrix of dimension dims.
    """

    return np.asarray(np.reshape(np.asarray(psi), dims)).T


def bell_state(qubits: int = 2) -> np.ndarray:
    """Create the maximally entangled state on dimension 2 ** qubits.

    :param qubits: Number of qubits.
    :return: The maximally entangled state on dimension 2 ** qubits.
    """
    return proj(vec(np.identity(qubits)))


a = np.array([[1, 1], [2, 2]])
b = np.identity(2)
c = np.array([[-1, 1], [0, 2]])
# print(np.kron(a,b) @ res(c))
# print( res(a @ c @ np.transpose(b)))

# print(bell_state(2))
# print(ketbra(vec(np.identity(2)/np.sqrt(2)), vec(np.identity(2)/np.sqrt(2))))

# pauli_matrices = np.array(([[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]))


# phi = np.pi
# print(np.array([1.0, phi]))

# print( [proj(np.identity(4)[i]) for i in range(4)])

# print(basis(4))

v = np.array([np.sqrt(2), 1j + 2])
x = np.array([np.pi, 1, -2j])
print(np.conjugate(x))
print(np.outer(v, np.conjugate(x)))
print(ketbra(v, x))
