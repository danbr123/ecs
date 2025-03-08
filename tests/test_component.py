# Tests Generated with ChatGPT
# TODO - improve tests, test edge cases

from ecs.component import Component
from ecs.array_wrapper import ArrayWrapper
from ecs.component import ComponentRegistry


# A dummy component subclass for testing.
class DummyComponent(Component):
    @property
    def dimensions(self) -> int:
        return 2


# ----------------------- Component Tests -----------------------


def test_component_initialization():
    comp = DummyComponent()
    # Check initial capacity (initial_capacity = 3 as defined in your code)
    assert comp._array.shape == (comp.initial_capacity, comp.dimensions)
    # Check that the underlying array is wrapped in an ArrayWrapper.
    assert isinstance(comp.array, ArrayWrapper)
    # Check that size is initially 0.
    assert comp.size == 0
    # Check default value.
    assert comp._default == (0, 0)


def test_component_add_and_get_value():
    comp = DummyComponent()
    # Add a new entity.
    comp.add(1, (10, 20))
    idx = comp.entity_to_index[1]
    # Ensure that value is stored correctly.
    val = comp.get_value(1)
    assert val == (10, 20)
    # Ensure size increments.
    assert comp.size == 1


def test_component_update_value():
    comp = DummyComponent()
    comp.add(1, (10, 20))
    comp.update_value(1, (100, 200))
    assert comp.get_value(1) == (100, 200)


def test_component_resize_and_reference_stability():
    comp = DummyComponent()
    initial_ref = comp.array
    # initial_capacity is 3; add 4 entities to force a resize.
    comp.add(1, (1, 2))
    comp.add(2, (3, 4))
    comp.add(3, (5, 6))
    comp.add(4, (7, 8))
    # Check that the ArrayWrapper reference is unchanged.
    assert comp.array is initial_ref
    # Verify the new entity's value.
    assert comp.get_value(4) == (7, 8)


def test_component_remove_and_free_slot():
    comp = DummyComponent()
    comp.add(1, (10, 20))
    comp.add(2, (30, 40))
    comp.add(3, (50, 60))
    # Remove an entity.
    comp.remove(2)
    # Assuming entity 2 was added second, its index should be freed.
    freed_indices = comp.free_slots
    assert len(freed_indices) >= 1
    # Now add a new entity; it should reuse a free slot.
    comp.add(4, (70, 80))
    reused_index = comp.entity_to_index[4]
    assert reused_index in freed_indices or reused_index < comp.size
    assert comp.get_value(4) == (70, 80)


# --------------------- ComponentRegistry Tests ---------------------


class DummyA(Component):
    @property
    def dimensions(self) -> int:
        return 2


class DummyB(Component):
    @property
    def dimensions(self) -> int:
        return 1


def test_registry_get_bit():
    registry = ComponentRegistry()
    bit_a = registry.get_bit(DummyA)
    bit_a_again = registry.get_bit(DummyA)
    assert bit_a == bit_a_again
    bit_b = registry.get_bit(DummyB)
    assert bit_a != bit_b


def test_registry_add_and_getitem():
    registry = ComponentRegistry()
    instance_a = DummyA()
    registry.add_component(DummyA, instance_a)
    assert DummyA in registry
    assert registry[DummyA] is instance_a


def test_registry_compute_signature_list_and_dict():
    registry = ComponentRegistry()
    # Clear any pre-existing state.
    registry._component_bits.clear()
    registry._next_bit = 1
    sig_list = registry.compute_signature([DummyA, DummyB])
    instance_a = DummyA()
    instance_b = DummyB()
    sig_dict = registry.compute_signature({DummyA: instance_a, DummyB: instance_b})
    assert sig_list == sig_dict
