from abc import abstractmethod, ABC
from numbers import Number
from typing import Tuple, Union, Optional, TypeVar, Dict, Type, List
import numpy as np

from ecs.array_wrapper import ArrayWrapper


class Component(ABC):
    """Base class for Components

    Uses numpy arrays to store components data.

    Allows efficient updates by performing batch updates using numpy operations on the
    entire entity array, instead of updating each entity separately.

    Since numpy arrays are stored in the same place in the memory - reduce memory cache
    misses and increases performance.

    Performing updates on this component can be done by modifying the `array` of the
    component in place, or overriding it with an array with a similar shape,
    e.g. `component.array = component.array * 3`
    The internal numpy array is wrapped with a stable ArrayWrapper, so it can be
    overridden with a new array without updating the references.
    in any override - the shape of the new array must match the shape of the original.

    Alternatively, entity data can be accessed directly using the `get_value` and
    `update_value` functions.

    Every component must inherit from this class in order to be processed properly.
    """

    initial_capacity = 3

    __slots__ = ("_array", "entity_to_index", "free_slots", "size")

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Return the number of elements stored per entity.
        For example, a 2D position would have dimensions == 2.
        """
        pass

    @property
    def _default(self) -> Tuple[Number, ...]:
        """Default value for new entities: a tuple of zeros by default."""
        if (default := getattr(self, "default", None)) is not None:
            return default
        return (0,) * self.dimensions

    def __init__(self) -> None:
        self._array = self._create_array(self.initial_capacity, self.dimensions)
        self.entity_to_index: dict[int, int] = {}
        self.free_slots: list[int] = []
        self.size: int = 0

    @staticmethod
    def _create_array(capacity: int, dims: int) -> ArrayWrapper:
        base_array = np.full((capacity, dims), np.nan)
        return ArrayWrapper(base_array)

    @property
    def array(self) -> ArrayWrapper:
        """Expose the underlying ArrayWrapper."""
        return self._array

    @array.setter
    def array(self, value: Union[np.ndarray, ArrayWrapper]):
        if not isinstance(value, (np.ndarray, ArrayWrapper)):
            raise TypeError("Assigned value must be a NumPy array or an ArrayWrapper.")
        expected_shape = (self._array.shape[0], self.dimensions)
        if value.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {value.shape}.")
        self._array.set_array(value)

    def add(
        self, entity_id: int, val: Optional[Union[Tuple[Number, ...], Number]] = None
    ) -> None:
        """Attach data for a new entity.

        The data is stored in a compact array slot. If free slots are available
        (from prior removals), one is reused; otherwise, data is appended at the end.
        The ArrayWrapper automatically ensures capacity if needed.
        """
        if val is None:
            val = self._default
        if isinstance(val, Number):
            val = (val,)
        if len(val) != self.dimensions:
            raise ValueError(
                f"Expected value with {self.dimensions} dimensions, got {len(val)}."
            )

        # Determine the index in the dense array.
        if self.free_slots:
            idx = self.free_slots.pop()
        else:
            idx = self.size
            if idx >= self._array.shape[0]:
                self._array.ensure_capacity(idx + 1)
        self._array[idx] = val
        self.entity_to_index[entity_id] = idx
        self.size += 1

    def remove(self, entity_id: int) -> None:
        """
        Remove an entity's data from the component.

        To maintain density, the data at the last active index is swapped into
        the removed slot, and the mapping is updated accordingly.
        The freed index is then added to the free list.
        """
        # TODO - when size reaches below a certain threshold - reorganize indexes and
        #   shrink array
        if entity_id not in self.entity_to_index:
            return
        idx = self.entity_to_index.pop(entity_id)
        last_index = self.size - 1
        if idx != last_index:
            # Find the entity that occupies the last active slot.
            swapped_entity = None
            for ent, pos in self.entity_to_index.items():
                if pos == last_index:
                    swapped_entity = ent
                    break
            if swapped_entity is not None:
                self._array[idx] = self._array[last_index]
                self.entity_to_index[swapped_entity] = idx
        self.free_slots.append(last_index)
        self.size -= 1

    def update_value(
        self, entity_id: int, val: Union[Tuple[Number, ...], Number]
    ) -> None:
        """
        Update the stored data for an entity.
        """
        if entity_id not in self.entity_to_index:
            raise ValueError("Entity not found.")
        if isinstance(val, Number):
            val = (val,)
        if len(val) != self.dimensions:
            raise ValueError(f"Expected {self.dimensions} dimensions, got {len(val)}.")
        idx = self.entity_to_index[entity_id]
        self._array[idx] = val

    def get_value(self, entity_id: int) -> Tuple[Number, ...]:
        """
        Retrieve the stored data for an entity.
        """
        if entity_id not in self.entity_to_index:
            raise ValueError("Entity not found.")
        idx = self.entity_to_index[entity_id]
        return tuple(self._array[idx])


_T = TypeVar("_T", bound=Component)
_CompDataT = Dict[Type[Component], _T]


class ComponentRegistry:
    """Stores components information per-world

    Allows a single instance per component type, and provides a function that extracts
    the signature of a list of components (signature may differ in different registries)
    """

    def __init__(self):
        self._component_bits: Dict[Type[Component], int] = {}
        self._next_bit = 1
        self.components: _CompDataT = {}

    def add_component(self, comp_type, instance):
        self.components[comp_type] = instance

    def get_bit(self, comp_type):
        if comp_type not in self._component_bits:
            self._component_bits[comp_type] = self._next_bit
            self._next_bit <<= 1
        return self._component_bits[comp_type]

    def compute_signature(
        self, components: Union[List[Type[Component]], _CompDataT]
    ) -> int:
        """Get unique signature for a composition of components.

        The signature is only relevant for a specific registry. Each may have a
        different signature for the same components list.

        Args:
            components: dictionary of component data - {type: instance} or list of types

        Returns:
            an integer that represents the signature of this component composition.
            Each component affects a unique bit in that signature.
        """
        signature = 0
        for comp_type in components:
            signature |= self.get_bit(comp_type)
        return signature

    def __contains__(self, key):
        return key in self.components

    def __setitem__(self, key, value):
        self.components[key] = value

    def __getitem__(self, item):
        return self.components[item]
