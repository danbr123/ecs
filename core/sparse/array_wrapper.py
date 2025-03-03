from typing import Union

import numpy as np


class ArrayWrapper:
    """Wraps np.ndarray to provide a stable reference when the array reference changes

    A stable wrapper around a NumPy array that forwards most operations
    (including arithmetic, bitwise, comparisons, and matrix multiplication)
    to the underlying array.

    This implementation uses __array_ufunc__ for ufunc calls and also
    explicitly defines most of the special methods so that Python's built-in
    operators (e.g., +, -, @) work as expected.
    """
    def __init__(self, array: np.ndarray):
        self._array = array

    def set_array(self, array: Union[np.ndarray, "ArrayWrapper"]) -> None:
        if isinstance(array, ArrayWrapper):
            array = array._array
        self._array = array

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(self, key, value):
        self._array[key] = value

    def __getattr__(self, attr):
        return getattr(self._array, attr)

    def __len__(self):
        return self._array.__len__()

    def __array__(self, *args, **kwargs):
        return self._array

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        new_inputs = [x._array if isinstance(x, ArrayWrapper) else x for x in inputs]
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        if isinstance(result, np.ndarray):
            return ArrayWrapper(result)
        return result

    # Arithmetic operators
    def __add__(self, other):
        return self._array + other

    def __radd__(self, other):
        return other + self._array

    def __sub__(self, other):
        return self._array - other

    def __rsub__(self, other):
        return other - self._array

    def __mul__(self, other):
        return self._array * other

    def __rmul__(self, other):
        return other * self._array

    def __truediv__(self, other):
        return self._array / other

    def __rtruediv__(self, other):
        return other / self._array

    def __floordiv__(self, other):
        return self._array // other

    def __rfloordiv__(self, other):
        return other // self._array

    def __mod__(self, other):
        return self._array % other

    def __rmod__(self, other):
        return other % self._array

    def __pow__(self, other, modulo=None):
        if modulo is None:
            return self._array ** other
        return pow(self._array, other, modulo)

    def __rpow__(self, other):
        return other ** self._array

    def __matmul__(self, other):
        return self._array @ other

    def __rmatmul__(self, other):
        return other @ self._array

    # Unary operators
    def __neg__(self):
        return -self._array

    def __pos__(self):
        return +self._array

    def __abs__(self):
        return abs(self._array)

    def __invert__(self):
        return ~self._array

    # Bitwise operators
    def __and__(self, other):
        return self._array & other

    def __rand__(self, other):
        return other & self._array

    def __or__(self, other):
        return self._array | other

    def __ror__(self, other):
        return other | self._array

    def __xor__(self, other):
        return self._array ^ other

    def __rxor__(self, other):
        return other ^ self._array

    def __lshift__(self, other):
        return self._array << other

    def __rlshift__(self, other):
        return other << self._array

    def __rshift__(self, other):
        return self._array >> other

    def __rrshift__(self, other):
        return other >> self._array

    # Comparison operators
    def __lt__(self, other):
        return self._array < other

    def __le__(self, other):
        return self._array <= other

    def __eq__(self, other):
        return self._array == other

    def __ne__(self, other):
        return self._array != other

    def __gt__(self, other):
        return self._array > other

    def __ge__(self, other):
        return self._array >= other

    def __repr__(self):
        return repr(self._array)
