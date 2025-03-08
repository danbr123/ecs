# Tests Generated with ChatGPT
# TODO - improve tests, test edge cases

import numpy as np

from ecs import Event
from ecs.world import World
from ecs.component import Component
from ecs.system import System


# Dummy components for testing.
class DummyA(Component):
    @property
    def dimensions(self) -> int:
        return 2


class DummyB(Component):
    @property
    def dimensions(self) -> int:
        return 1


# Dummy system that records update calls.
class DummySystem(System):
    group = "dummy"

    def __init__(self):
        super().__init__()
        self.updates: list[float] = []

    def update(self, world: World, dt: float) -> None:
        self.updates.append(dt)


def test_create_entity():
    world = World()
    # Register dummy components.
    world.register_component(DummyA)
    world.register_component(DummyB)
    # Create entity with both components.
    entity_id = world.create_entity(
        [DummyA, DummyB], component_initial_data={DummyA: (10, 20), DummyB: (30,)}
    )
    # Check that the entity mapping exists.
    assert entity_id in world.entity_components
    comp_map = world.entity_components[entity_id]
    assert DummyA in comp_map
    assert DummyB in comp_map

    # Query the entity.
    results = world.query([DummyA, DummyB])
    assert len(results) == 1
    q_entity, q_map = results[0]
    assert q_entity == entity_id

    # Verify stored values using the component instances.
    compA = world.get_component_instance(DummyA)
    compB = world.get_component_instance(DummyB)
    idxA = compA.entity_to_index.get(entity_id)
    idxB = compB.entity_to_index.get(entity_id)
    np.testing.assert_array_equal(compA.array[idxA], np.array((10, 20)))
    np.testing.assert_array_equal(compB.array[idxB], np.array((30,)))


def test_remove_entity():
    world = World()
    world.register_component(DummyA)
    entity_id = world.create_entity([DummyA], component_initial_data={DummyA: (5, 6)})
    assert entity_id in world.entity_components
    world.remove_entity(entity_id)
    assert entity_id not in world.entity_components
    assert entity_id in world.free_ids


def test_add_remove_component():
    world = World()
    world.register_component(DummyA)
    world.register_component(DummyB)
    entity_id = world.create_entity([DummyA], component_initial_data={DummyA: (1, 2)})
    comp_map = world.entity_components[entity_id]
    assert DummyA in comp_map
    # Add DummyB.
    world.add_component(entity_id, DummyB, (3,))
    comp_map = world.entity_components[entity_id]
    assert DummyB in comp_map
    # Remove DummyA.
    world.remove_component(entity_id, DummyA)
    comp_map = world.entity_components[entity_id]
    assert DummyA not in comp_map


def test_query_cache_invalidation():
    world = World()
    world.register_component(DummyA)
    entity_id = world.create_entity([DummyA], component_initial_data={DummyA: (7, 8)})
    # First query.
    results1 = world.query([DummyA])
    # Remove entity, which should invalidate the cache.
    world.remove_entity(entity_id)
    results2 = world.query([DummyA])
    # The first query should have one result; the second should have none.
    assert len(results1) == 1
    assert len(results2) == 0


def test_update_systems():
    world = World()
    dummy = DummySystem()
    world.register_system(dummy)
    world.update_systems(0.016, group="dummy")
    assert len(dummy.updates) == 1
    assert dummy.updates[0] == 0.016


def test_event_bus_integration():
    # Simple test to ensure the world's event bus is accessible and updates.
    world = World()

    # Publish a dummy event synchronously via event bus.
    class DummyEvent(Event):
        pass

    # Temporarily attach a flag.
    flag = {"called": False}

    def handler(event):
        flag["called"] = True

    world.event_bus.subscribe(DummyEvent, handler)
    world.event_bus.publish_async(DummyEvent())
    # Before update, flag should be False.
    assert not flag["called"]
    world.event_bus.update()
    assert flag["called"]
