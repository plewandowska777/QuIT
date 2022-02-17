import numpy as np
import pytest
from scipy import linalg
from scipy.linalg import sqrtm
from scipy.stats import unitary_group

from quit.basicfunction import bell_state, braket, dagger, ket, proj
from quit.distance import (
    fidelity,
    frobenius_norm,
    sqrt_fidelity,
    superfidelity,
    trace_distance,
    trace_norm,
)


@pytest.mark.parametrize("rho", [proj([1, 2]), proj([1, 0])])
@pytest.mark.parametrize("sigma", [proj([0, 1]), proj([1, np.pi])])
def test_trace_distance_for_projectors(rho, sigma):
    np.testing.assert_almost_equal(
        trace_distance(rho, sigma), 2 * np.sqrt(1 - np.abs(braket(rho, sigma)))
    )


@pytest.mark.parametrize("matrix", [np.identity(3), bell_state(4), linalg.hadamard(4)])
def test_frobenius_norm_from_definition(matrix):
    np.testing.assert_equal(frobenius_norm(matrix), np.sqrt(np.trace(dagger(matrix) @ matrix)))


@pytest.mark.parametrize("matrix", [np.identity(3), bell_state(4), linalg.hadamard(4)])
def test_frobenius_norm_as_sum_entries(matrix):
    np.testing.assert_equal(frobenius_norm(matrix), np.sqrt(np.sum(np.abs(matrix) ** 2)))


@pytest.mark.parametrize("rho", [np.identity(2), proj([1, 0])])
@pytest.mark.parametrize("sigma", [np.diag([1, 2]), proj([0, 1])])
@pytest.mark.parametrize("U", [1 / np.sqrt(2) * linalg.hadamard(2), unitary_group.rvs(2)])
def test_sqrt_fidelity_function_unitary_invariance(rho, sigma, U):
    np.testing.assert_almost_equal(
        sqrt_fidelity(rho, sigma), sqrt_fidelity(U @ rho @ dagger(U), U @ sigma @ dagger(U))
    )


@pytest.mark.parametrize("rho", [np.identity(2), proj(ket([1, 0]))])
@pytest.mark.parametrize("sigma", [np.identity(2), proj(ket([1, 0]))])
def test_fidelity_function_for_qubits(rho, sigma):
    np.testing.assert_almost_equal(
        fidelity(rho, sigma),
        np.trace(rho @ sigma) + 2 * np.sqrt(linalg.det(rho) * linalg.det(sigma)),
    )


@pytest.mark.parametrize("rho", [np.identity(2), proj(ket([1, 0]))])
@pytest.mark.parametrize("sigma", [np.identity(2), proj(ket([1, 0]))])
def test_fidelity_function_by_using_trace_norm(rho, sigma):
    np.testing.assert_almost_equal(fidelity(rho, sigma), trace_norm(sqrtm(rho) @ sqrtm(sigma)) ** 2)


@pytest.mark.parametrize("rho", [np.diag([1, 2, 3])])
@pytest.mark.parametrize("sigma", [np.diag([4, 5, 6])])
def test_fidelity_function_if_matrices_commute(rho, sigma):
    np.testing.assert_almost_equal(fidelity(rho, sigma), np.trace(sqrtm(rho @ sigma)) ** 2)


@pytest.mark.parametrize("rho", [1 / 2 * np.identity(2), proj(ket([1, 0]))])
@pytest.mark.parametrize("sigma", [1 / 2 * np.identity(2), proj(ket([1, 0]))])
@pytest.mark.parametrize("U", [1 / np.sqrt(2) * linalg.hadamard(2)])
def test_superfidelity_function_unitary_invariance(rho, sigma, U):
    np.testing.assert_almost_equal(
        superfidelity(rho, sigma), superfidelity(U @ rho @ dagger(U), U @ sigma @ dagger(U))
    )


@pytest.mark.parametrize("rho", [1 / 2 * np.identity(2), proj(ket([1, 0]))])
@pytest.mark.parametrize("sigma", [1 / 2 * np.identity(2), proj(ket([1, 0]))])
def test_superfidelity_as_upper_bound_of_fidelity(rho, sigma):
    assert fidelity(rho, sigma) - superfidelity(rho, sigma) <= 10 ** (-5)
