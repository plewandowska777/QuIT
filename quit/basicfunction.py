import typing

import numpy as np
from scipy import linalg

Array1D = typing.Union[np.ndarray, typing.Sequence[complex]]


def ket(psi: Array1D) -> np.ndarray:
    """Create a ket |psi> as a vertical vector.

    :param psi: Sequence of complex numbers.
    :return: A vertical vector as numpy array.
    """
    psi = np.asarray(psi)

    if len(psi.squeeze().shape) != 1:
        raise ValueError("Wrong parameter psi.")
    return psi.reshape((-1, 1))


def basis(dim: int, index: int) -> np.ndarray:
    """Create computational basis of dimension dim.

    :param dim: Dimension of vector space.
    :param index:
    :return: A basis vector |d, i>.
    """
    if index >= dim:
        raise Exception("Parameter index out of range.")

    vec = np.zeros(dim)
    vec[index] = 1
    return vec


def dagger(matrix: np.ndarray) -> np.ndarray:
    """Create conjugate transpose of a given matrix.

    :param matrix: A matrix given by numpy array.
    :return: An operator dagger as concatanation conjugate and transpose.
    """
    return np.transpose(np.conjugate(matrix))


def bra(psi: Array1D) -> np.ndarray:
    """Create bra <psi| as a horizontal vector.

    :param psi: Sequence of complex numbers.
    :return: A horizontal vector as numpy array satisfying <psi| = |psi>^+
    """
    return dagger(ket(psi))


def braket(psi: Array1D, phi: Array1D) -> complex:
    """Create braket as an inner product between vectors psi and phi.

    :param psi: Sequence of complex numbers.
    :param phi: Sequence of complex numbers.
    :return: Inner product between vectors psi and phi.
    """
    return complex(bra(psi) @ ket(phi))


def ketbra(psi: Array1D, phi: Array1D) -> np.ndarray:
    """Create a matrix from multiplication vectors psi and phi.

    :param psi: Sequence of complex numbers.
    :param phi: Sequence of complex numbers.
    :return: A matrix |psi><phi|
    """
    return ket(psi) @ bra(phi)


def proj(psi: Array1D) -> np.ndarray:
    """Create a projector ontto normalized vector |psi>.

    :param psi: Sequence of complex numbers.
    :return: A matrix |psi><psi|.
    """
    return ket(psi) / linalg.norm(psi) @ bra(psi) / linalg.norm(psi)


def res(matrix: np.ndarray) -> np.ndarray:
    """Reshaping maps matrix into a vector row by row. res(A) is equivalent to vec(A.T).

    :param matrix: A matrix given by numpy array.
    :return: A vertical vector creating by matrix reshaping.
    """
    return ket(np.reshape(np.asarray(matrix), np.size(matrix)))


def unres(psi: Array1D, dims: typing.Tuple[int, int]) -> np.ndarray:
    """Un-reshaping of the vector into the matrix of dimension dims.

    :param psi: A matrix given by numpy array.
    :param dims: A tuple of dimensions.
    :return: A vertical vector creating by matrix vectorization
    """
    return np.reshape(np.asarray(psi), dims)


def vec(matrix: np.ndarray) -> np.ndarray:
    """Reshaping maps matrix into a vector column by column. vec(A) is equivalent to res(A.T).

    :param matrix: A matrix given by numpy array.
    :return: A matrix of dimension dims.
    """
    return ket(np.reshape(np.asarray(matrix).T, np.size(matrix)))


def unvec(psi: Array1D, dims: typing.Tuple[int, int]) -> np.ndarray:
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
    return vec(np.identity(qubits)) / linalg.norm(vec(np.identity(qubits)))


def bell_state_density(qubits: int = 2) -> np.ndarray:
    """Create the maximally entangled state on dimension 2 ** qubits as density operator.

    :param qubits: Number of qubits.
    :return: The maximally entangled state on dimension 2 ** qubits as density operator.
    """
    return proj(vec(np.identity(qubits)))
