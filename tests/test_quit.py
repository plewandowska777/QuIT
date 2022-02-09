import numpy as np
import pytest

from quit.basicfunction import base, bra, braket, ket, ketbra, max_ent, proj, res, vec


def test_base():
    np.testing.assert_array_equal(base(0, 2), np.asarray([[1], [0]]))


def test_base_by_using_ket():
    np.testing.assert_array_equal(base(0, 2), ket([1, 0]))


def test_ket_by_using_base():
    np.testing.assert_array_equal(base(0, 2), ket(np.asarray([1, 0])))


@pytest.mark.parametrize("phi", [np.pi, np.sqrt(2), 2j, 1 + 1j, 1])
def test_ket_different_types(phi):
    np.testing.assert_array_equal(ket(np.asarray([1.0, phi])), ket(np.asarray([1, phi])))


def test_ket_types_float_complex():
    np.testing.assert_array_equal(ket(np.asarray([1.0, 0.0 + 1.0j])), ket(np.asarray([1, 1j])))


def test_proj_by_using_base():
    np.testing.assert_array_equal(base(0, 3) @ bra(base(0, 3)), proj(np.asarray([1, 0, 0])))


@pytest.mark.parametrize("phi", [np.sqrt(2), 1j + 2])
@pytest.mark.parametrize("psi", [np.pi, 1, 2j])
def test_braket(phi, psi):
    assert braket(phi, psi) - np.inner(np.conjugate(phi), psi) == 0


def test_ketbra_by_using_proj():
    np.testing.assert_array_equal(ketbra([1, 0, 0], [1, 0, 0]), proj(np.asarray([1, 0, 0])))


def test_ketbra_by_using_base():
    np.testing.assert_array_equal(ketbra([0, 1, 0], [1, 0, 0]), base(1, 3) @ bra(base(0, 3)))


matrix = np.array([[1, 2j, 1 - 1j], [0, 1, 2], [np.pi, 0, -1]])


def test_function_res():
    np.testing.assert_array_equal(
        res(matrix), ket(np.asarray([1, 2j, 1 - 1j, 0, 1, 2, np.pi, 0, -1]))
    )


def test_function_vec():
    np.testing.assert_array_equal(
        vec(matrix), ket(np.asarray([1, 0, np.pi, 2j, 1, 0, 1 - 1j, 2, -1]))
    )


def test_if_vec_and_res_is_equal_on_symmetric_matrix():
    np.testing.assert_array_equal(vec(np.identity(3)), res(np.identity(3)))


@pytest.mark.parametrize("dim", [2, 3, 4])
def max_ent_state(dim):
    np.testing.assert_array_equal(
        max_ent(dim), 1 / np.sqrt(dim) * ketbra(vec(np.identity(dim)), vec(np.identity(dim)))
    )
