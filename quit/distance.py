import numpy as np
from scipy import linalg
from scipy.linalg import sqrtm, svdvals

from quit.basicfunction import res


def trace_norm(rho: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/trace-norm -
    Trace norm is the sum of the singular values of matrix rho.

    :param rho: A matrix as numpy array.
    :return: Trace norm of matrix rho (non-negative number).
    """
    if rho.shape[0] != rho.shape[1]:
        raise Exception("Non square matrix")
    return np.sum(svdvals(rho))


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> complex:
    """https://www.quantiki.org/wiki/trace-distance -
    The trace distance (also called the variational or Kolmogorov distance)
    between operators rho and sigma.

    :param rho: A squere matrix az numpy array.
    :param sigma: A squere matrix az numpy array.
    :return: Distance (trace norm) between operators rho and sigma (non-negative number).
    """
    if rho.shape[0] != rho.shape[1]:
        raise Exception("Non square matrix")

    if sigma.shape[0] != sigma.shape[1]:
        raise Exception("Non square matrix")

    if rho.shape[0] != sigma.shape[0]:
        raise Exception("Matrices are of different dimensions.")

    return trace_norm(np.subtract(rho, sigma))
    # return 1/2 * np.trace(sqrtm( dagger(np.subtract(rho, sigma)) @ np.subtract(rho, sigma) ))


def frobenius_norm(rho: np.ndarray) -> float:
    """https://mathworld.wolfram.com/FrobeniusNorm.html -
    Create Frobenius norm of matrix rho.

    :param rho: A matrix rho as numpy array.
    :return:  Frobenius norm of operator rho (non-negative number).
    """
    return linalg.norm(res(rho))


def sqrt_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/Fidelity -
    Measure between quantum states (more generally for positive semidefinite operators)

    :param rho: A density operator rho as numpy array.
    :param sigma: A density operator rho as numpy array.
    :return: Distance (square fidelity) between operators rho and sigma (non-negative number).
    """
    if rho.shape[0] != rho.shape[1]:
        raise Exception("Non square matrix")

    if sigma.shape[0] != sigma.shape[1]:
        raise Exception("Non square matrix")

    if rho.shape[0] != sigma.shape[0]:
        raise Exception("Matrices are of different dimensions.")

    return np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho)))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """https://www.quantiki.org/wiki/Fidelity -
    Measure between quantum states (more generally for positive semidefinite operators).

    :param rho: A density operator rho as numpy array.
    :param sigma: A density operator rho as numpy array.
    :return: Distance (fidelity) between operators rho and sigma (non-negative number).
    """

    if rho.shape[0] != rho.shape[1]:
        raise Exception("Non square matrix")

    if sigma.shape[0] != sigma.shape[1]:
        raise Exception("Non square matrix")

    if rho.shape[0] != sigma.shape[0]:
        raise Exception("Matrices are of different dimensions.")

    return sqrt_fidelity(rho, sigma) ** 2


def superfidelity(rho: np.ndarray, sigma: np.ndarray) -> complex:
    """https://www.quantiki.org/wiki/superfidelity -
    A measure of similarity between density operators.

    :param rho: A density operator rho as numpy array.
    :param sigma: A density operator rho as numpy array.
    :return: Distance (superfidelity) between operators rho and sigma (non-negative number).
    """
    return np.trace(rho @ sigma) + np.sqrt(1 - np.trace(rho @ rho)) * np.sqrt(
        1 - np.trace(sigma @ sigma)
    )
