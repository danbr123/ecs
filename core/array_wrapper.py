from typing import Union
import numpy as np


class ArrayWrapper:
    """Wraps np.ndarray to provide a stable reference when the array reference changes.

    A stable wrapper around a NumPy array that forwards most operations
    (including arithmetic, bitwise, comparisons, and matrix multiplication)
    to the underlying array.

    This implementation uses __array_ufunc__ for ufunc calls and also
    explicitly defines most of the special methods so that Python's built-in
    operators (e.g., +, -, @) work as expected.
    """
    scale_factor = 1.5

    def __init__(self, array: np.ndarray):
        self._array = array

    def set_array(self, array: Union[np.ndarray, "ArrayWrapper"]) -> None:
        if isinstance(array, ArrayWrapper):
            array = array._array
        self._array = array

    def ensure_capacity(self, min_rows: int) -> None:
        """Ensure that the underlying array has at least `min_rows` rows.

        If the current number of rows is less than `min_rows`, the array is resized.
        The new capacity is the maximum of min_rows and twice the current row count.
        New slots are filled with np.nan.
        """
        current_rows, current_cols = self._array.shape
        if min_rows <= current_rows:
            return
        new_rows = max(min_rows, current_rows * self.scale_factor)
        new_array = np.full((new_rows, current_cols), np.nan, dtype=self._array.dtype)
        new_array[:current_rows] = self._array
        self.set_array(new_array)

    def shrink_to(self, new_rows: int) -> None:
        """Shrink the underlying array to exactly new_rows rows.

        This method copies the first new_rows rows into a new array.
        Use with care: frequent shrinking can lead to performance issues.
        """
        current_rows, current_cols = self._array.shape
        if new_rows >= current_rows:
            return
        new_array = np.copy(self._array[:new_rows])
        self.set_array(new_array)

    def __getitem__(self, key):
        return self._array[key]

    def __setitem__(self, key, value):
        self._array[key] = value

    def __getattr__(self, attr):
        return getattr(self._array, attr)

    def __len__(self):
        return len(self._array)

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
