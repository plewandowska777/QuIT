import numpy as np

from quit.basicfunction import base, bra, ket, proj


def test_base():
    assert all(base(0, 2) == np.asarray([[1], [0]], dtype=complex))


def test_ket():
    assert all(base(0, 2) == ket([1, 0]))


def test_proj():
    assert all(base(0, 3) @ bra(base(0, 3)) == proj([0, 1, 0]))
