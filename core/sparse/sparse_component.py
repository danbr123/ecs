from abc import abstractmethod
from numbers import Number
from typing import Tuple, Union, Optional

import numpy as np

from core.component import Component
from core.utils import Singleton
from core.sparse.array_wrapper import ArrayWrapper


class SparseComponent(Component, metaclass=Singleton):
    """Store entity information using a numpy array

    IMPORTANT: Each class represents a single component shared by ALL entities (entities
    that do not use the component will have null values in the array).
    Only one component can exist per class, so in order to create multiple sparse
    component this class must be subclassed.

    Allows efficient updates for simple and common components such as position, by
    performing batch updates using numpy operations on the entire entity array, instead
    of updating each entity separately.

    Performing updates on this component can be done by modifying the `array` of the
    component in place, or overriding it with an array with a similar shape,
    e.g. `component.array = component.array * 3`
    The internal numpy array is wrapped with a stable ArrayWrapper, so it can be
    overridden with a new array without updating the references.
    in any override - the shape of the new array must match the shape of the original.
    """
    sparse = True
    initial_array_size = 1000

    __slots__ = "_array"

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Size of the per-entity data in the component - affects array shape"""
        pass

    @property
    def default_val(self) -> Tuple[Number, ...]:
        """Default value for new entities when no value is specified"""
        return (0, ) * self.dimensions

    def __init__(self) -> None:
        self._array = self._create_array(self.initial_array_size, self.dimensions)

    def _create_array(self, rows: int, cols: int) -> ArrayWrapper:
        base_array = np.full((rows, cols), np.nan)
        return ArrayWrapper(base_array)

    def add(
            self,
            entity_id: int,
            val: Optional[Union[Tuple[Number, ...], Number]] = None
    ) -> None:
        """Attach the component to a new entity

        Update the array at position `entity_id` with data provided by the user.
        If no data is provided, use the default data.

        Args:
            entity_id (int): entity to attach the component to
            val (Tuple): initial component data for the entity - must match the component
                dimension
        """
        if val is None:
            val = self.default_val
        if isinstance(val, Number):
            val = (val,)
        if len(val) != self.dimensions:
            raise ValueError(
                f"Expected value with {self.dimensions} dimensions, got {len(val)}."
            )
        try:
            self.array[entity_id] = val
        except IndexError:
            self.update_size()
            self.array[entity_id] = val

    def remove(self, entity_id: int) -> None:
        """Detach the component from an entity.

        Invalidate the slot for the given entity by filling the slot with NaNs.

        Args:
            entity_id (int): entity to remove from the component's array
        """
        try:
            self.array[entity_id] = (np.nan,) * self.dimensions
        except IndexError:
            pass

    def update_size(self) -> None:
        """Double the array size while preserving shape and type."""
        current = self._array
        new_size = current.shape[0] * 2
        new_array = np.empty((new_size, current.shape[1]), dtype=current.dtype)
        new_array[: current.shape[0]] = current
        new_array[current.shape[0]:].fill(np.nan)
        self._array.set_array(new_array)

    @property
    def array(self) -> ArrayWrapper:
        """Expose the internal ArrayWrapper."""
        return self._array

    @array.setter
    def array(self, value: np.ndarray):
        if not isinstance(value, (np.ndarray, ArrayWrapper)):
            raise TypeError("Assigned value must be a NumPy array or an ArrayWrapper.")
        if value.shape != self._array.shape:
            raise ValueError(
                f"Cannot change shape from {self._array.shape} to {value.shape}.")
        self._array.set_array(value)
