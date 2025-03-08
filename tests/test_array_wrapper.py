# Tests Generated with ChatGPT
# TODO - improve tests, test edge cases
import numpy as np
import pytest

from ecs.array_wrapper import ArrayWrapper


@pytest.fixture
def base_array():
    return np.array([[1, 2], [3, 4], [5, 6]], dtype=float)


@pytest.fixture
def wrapper(base_array):
    return ArrayWrapper(base_array)


def test_initial_state(wrapper, base_array):
    np.testing.assert_array_equal(wrapper._array, base_array)
    assert len(wrapper) == len(base_array)


def test_set_array(wrapper):
    new_array = np.array([[7, 8], [9, 10]], dtype=float)
    wrapper.set_array(new_array)
    np.testing.assert_array_equal(wrapper._array, new_array)
    other = ArrayWrapper(np.array([[11, 12]], dtype=float))
    wrapper.set_array(other)
    np.testing.assert_array_equal(wrapper._array, other._array)


def test_ensure_capacity(wrapper):
    orig_shape = wrapper._array.shape
    wrapper.ensure_capacity(orig_shape[0] + 10)
    new_shape = wrapper._array.shape
    assert new_shape[0] >= orig_shape[0] + 10
    np.testing.assert_array_equal(
        wrapper._array[: orig_shape[0]], np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
    )


def test_shrink_to(wrapper):
    wrapper.ensure_capacity(10)
    for i in range(5):
        wrapper._array[i] = np.array([i, i + 1], dtype=float)
    wrapper.shrink_to(5)
    assert wrapper._array.shape[0] == 5
    expected = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]], dtype=float)
    np.testing.assert_array_equal(wrapper._array, expected)


def test_arithmetic(wrapper):
    result_add = wrapper + 10
    expected_add = wrapper._array + 10
    np.testing.assert_array_equal(result_add, expected_add)

    result_sub = 10 - wrapper
    expected_sub = 10 - wrapper._array
    np.testing.assert_array_equal(result_sub, expected_sub)

    result_mul = wrapper * 2
    expected_mul = wrapper._array * 2
    np.testing.assert_array_equal(result_mul, expected_mul)


def test_array_ufunc(wrapper):
    result = np.add(wrapper, 5)
    assert isinstance(result, ArrayWrapper)
    expected = wrapper._array + 5
    np.testing.assert_array_equal(result._array, expected)


def test_comparison(wrapper):
    assert (wrapper < 10).all() == (wrapper._array < 10).all()
    np.testing.assert_array_equal(
        wrapper == wrapper._array, wrapper._array == wrapper._array
    )


def test_getitem_setitem(wrapper):
    val = wrapper[1]
    np.testing.assert_array_equal(val, wrapper._array[1])
    wrapper[1] = [100, 200]
    np.testing.assert_array_equal(wrapper._array[1], np.array([100, 200]))
