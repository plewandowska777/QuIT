import numpy as np
import pytest
from scipy import linalg

from quit.basicfunction import (
    basis,
    bell_state,
    bra,
    braket,
    dagger,
    ket,
    ketbra,
    proj,
    res,
    unres,
    unvec,
    vec,
)


@pytest.mark.parametrize("phi", [np.pi, np.sqrt(2), 2j, 1 + 1j, 1])
def test_ket_with_different_types_of_entries(phi):
    np.testing.assert_array_equal(np.array([[1.0], [phi]]), ket([1, phi]))


@pytest.mark.parametrize("dim", [2, 4])
def test_if_basis_is_correctly_defined(dim):
    comp_basis = [np.identity(dim)[i] for i in range(dim)]
    np.testing.assert_array_equal(basis(dim), comp_basis)


@pytest.mark.parametrize("phi", [np.pi, np.sqrt(2), 2j, 1 + 1j, 1])
def test_bra_with_different_types_of_entries(phi):
    np.testing.assert_array_equal(np.array([[1, np.conjugate(phi)]]), bra([1, phi]))


@pytest.mark.parametrize("phi", [[np.sqrt(2), 1j + 2]])
@pytest.mark.parametrize("psi", [[np.pi, 1, 2j]])
def test_ketbra_is_equal_outer_product(phi, psi):
    np.testing.assert_almost_equal(ketbra(phi, psi), np.outer(phi, np.conjugate(psi)))


@pytest.mark.parametrize("phi", [np.sqrt(2), 1j + 2])
@pytest.mark.parametrize("psi", [np.pi, 1, 2j])
def test_braket(phi, psi):
    assert braket(phi, psi) - np.inner(np.conjugate(phi), psi) == 0


@pytest.mark.parametrize("vector", [[1, 2], [1, -1]])
def test_projector_is_equal_proj_to_the_power_of_2(vector):
    np.testing.assert_almost_equal(proj(vector) @ proj(vector), proj(vector))


sx = np.array([[0, 1], [1, 0]])


@pytest.mark.parametrize(
    "symmatrix", [np.identity(3), np.asarray(linalg.hadamard(2, dtype=complex)), np.kron(sx, sx)]
)
def test_dagger_for_hermitian_matrices(symmatrix):
    np.testing.assert_array_equal(dagger(symmatrix), symmatrix)


matrix = np.array([[1, 1 - 1j], [0, 1], [np.pi, -1]])


def test_function_res_naive():
    np.testing.assert_array_equal(res(matrix), ket(np.asarray([1, 1 - 1j, 0, 1, np.pi, -1])))


def test_function_res_by_using_telegraphic_notation():

    a = np.array([[1, 1], [2, 2]])
    b = np.identity(2)
    c = np.array([[-1, 1], [0, 2]])

    np.testing.assert_array_equal(np.kron(a, b) @ res(c), res(a @ c @ np.transpose(b)))


def test_function_res_for_rank_one_operator():
    x = [1, 1 - 1j]
    y = [np.pi, -1]
    np.testing.assert_array_equal(res(ketbra(x, y)), ket(np.kron(x, y)))


def test_function_res_for_vector():
    x = [1, 1 - 1j, np.pi, -1]
    np.testing.assert_array_equal(res(x), ket(x))


def test_function_vec_naive():
    np.testing.assert_array_equal(vec(matrix), ket(np.asarray([1, 0, np.pi, 1 - 1j, 1, -1])))


def test_function_vec_as_transposition_of_res():
    np.testing.assert_array_equal(vec(matrix), res(np.transpose(matrix)))


sx = np.array([[0, 1], [1, 0]])


@pytest.mark.parametrize(
    "symmatrix", [np.identity(3), np.asarray(linalg.hadamard(2, dtype=complex)), np.kron(sx, sx)]
)
def test_if_vec_and_res_is_equal_on_symmetric_matrix(symmatrix):
    np.testing.assert_array_equal(vec(symmatrix), res(symmatrix))


vector = np.array([[1, 1 - 1j, 0, 1, np.pi, -1]])
matrix = np.array([[1, 1 - 1j], [0, 1], [np.pi, -1]])


def test_unres():
    np.testing.assert_array_equal(unres(vector, (3, 2)), matrix)


def test_unvec():
    np.testing.assert_array_equal(unvec(vector, (3, 2)), matrix.transpose())


@pytest.mark.parametrize("dim", [2, 3, 4])
def test_bell_state(dim):
    np.testing.assert_array_equal(
        bell_state(dim),
        ketbra(vec(np.identity(dim) / np.sqrt(dim)), vec(np.identity(dim) / np.sqrt(dim))),
    )
