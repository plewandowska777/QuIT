import numpy as np
from scipy import linalg
from scipy.linalg import eigvals, sqrtm, svdvals

from quit.basicfunction import dagger, res


def if_operator_hermitian(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, dagger(matrix))


def if_operator_positive(matrix: np.ndarray) -> bool:

    if if_operator_hermitian(matrix) is not True:
        return False
    for eigen in eigvals(matrix):
        if eigen < -(10 ** (-5)):
            return False
    return True


def if_operator_density(matrix: np.ndarray) -> bool:
    return if_operator_positive(matrix) and np.allclose(np.trace(matrix), 1)


def trace_norm(rho: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/trace-norm -
    Trace norm is the sum of the singular values of matrix rho.

    :param rho: A matrix as numpy array.
    :return: Trace norm of matrix rho (non-negative number).
    """

    return np.sum(svdvals(rho))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/trace-distance -
    The trace distance (also called the variational or Kolmogorov distance)
    between two density operators rho and sigma.

    :param rho: A density operator.
    :param sigma: A density operator.
    :return: Distance (trace norm) between density operators rho and sigma (non-negative number).
    """
    if (if_operator_density(rho) is not True) or (if_operator_density(sigma) is not True):
        raise Exception("The operators are not density operators.")
    return trace_norm(np.subtract(rho, sigma))


def frobenius_norm(rho: np.ndarray) -> float:
    """https://mathworld.wolfram.com/FrobeniusNorm.html -
    Create Frobenius norm of matrix rho.

    :param rho: A matrix as numpy array.
    :return:  Frobenius norm of operator rho (non-negative number).
    """
    return linalg.norm(res(rho))


def sqrt_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/Fidelity -
    Measure between positive semidefinite operators.

    :param rho: A positive semidefinite operator as numpy array.
    :param sigma: A positive semidefinite operator as numpy array.
    :return: Distance (square fidelity) between two positive semidefinite operators
    (non-negative number).
    """
    if (if_operator_positive(rho) is not True) or (if_operator_positive(sigma) is not True):
        raise Exception("The operators are not positive semidefinite.")

    return np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/Fidelity -
    Measure between positive semidefinite operators.

    :param rho: A positive semidefinite operator as numpy array.
    :param sigma: A positive semidefinite operator as numpy array.
    :return: Distance (fidelity) between two positive semidefinite operators (non-negative number).
    """

    if (if_operator_positive(rho) is not True) or (if_operator_positive(sigma) is not True):
        raise Exception("The operators are not positive semidefinite.")

    return sqrt_fidelity(rho, sigma) ** 2


def superfidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/superfidelity -
    A measure of similarity between density operators.

    :param rho: A density operator rho as numpy array.
    :param sigma: A density operator rho as numpy array.
    :return: Distance (superfidelity) between operators rho and sigma (non-negative number).
    """
    if (if_operator_density(rho) is not True) or (if_operator_density(sigma) is not True):
        raise Exception("The operators are not density operators.")

    return np.trace(rho @ sigma) + np.sqrt(np.abs(1 - np.trace(rho @ rho))) * np.sqrt(
        np.abs(1 - np.trace(sigma @ sigma))
    )
