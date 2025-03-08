from numbers import Number
from typing import Dict, List, Optional, Tuple, Union, TypeVar, Type
import warnings

from ecs.system import System
from ecs.component import Component, ComponentRegistry
from ecs.event import EventBus

_T = TypeVar("_T", bound=Component)
_CompDataT = Dict[Type[Component], _T]


class Archetype:
    """An archetype groups entities that share the same component composition.

    It stores:
      - signature: an integer bitmask representing the set of component types.
      - entities: a list of entity IDs.
      - storage: a dictionary mapping component types to a list of component data.
      - index_map: mapping from entity ID to its index in the storage arrays.

    Removal uses a swap‐and‐pop strategy for efficiency.
    """
    __slots__ = ('signature', 'entities', 'storage', 'index_map')

    def __init__(self, signature: int) -> None:
        self.signature: int = signature
        self.entities: List[int] = []
        self.storage: Dict[Type[Component], List[Component]] = {}
        self.index_map: Dict[int, int] = {}  # entity_id -> index

    def add_entity(self, entity_id: int, components: _CompDataT) -> None:
        """Add an entity along with its component data.

        Assumes that the keys of `components` match the archetype's composition.

        Args:
            entity_id (int): id of the entity to add
            components (_CompDataT): Dictionary of {type: instance} with the components
                of that entity.
        """
        index = len(self.entities)
        self.entities.append(entity_id)
        self.index_map[entity_id] = index
        for comp_type, comp_instance in components.items():
            if comp_type not in self.storage:
                self.storage[comp_type] = []
            self.storage[comp_type].append(comp_instance)

    def remove_entity(self, entity_id: int) -> Optional[_CompDataT]:
        """Remove an entity using swap‐and‐pop.

        Returns the removed component data (for potential reuse) or None if not found.

        Args:
            entity_id (int): id of the entity to remove
        """
        if entity_id not in self.index_map:
            return None
        index = self.index_map[entity_id]
        last_index = len(self.entities) - 1
        removed_data: _CompDataT = {}
        for comp_type, data_list in self.storage.items():
            removed_data[comp_type] = data_list[index]
        if index != last_index:
            last_entity = self.entities[last_index]
            self.entities[index] = last_entity
            self.index_map[last_entity] = index
            for comp_type, data_list in self.storage.items():
                data_list[index] = data_list[last_index]
        self.entities.pop()
        for comp_type, data_list in self.storage.items():
            data_list.pop()
        del self.index_map[entity_id]
        return removed_data


class World:
    """Main class of the ECS framework.

    The ECS World manages entities, components, archetypes, and systems.

    It supports:
      - Archetype‐based storage for components (using bit masks).
      - A system update loop.
      - Query caching to speed up repeated queries.
      - Efficient cleanup via free lists and swap‐and‐pop removal.
    """
    def __init__(self) -> None:
        self.archetypes: Dict[int, Archetype] = {}
        self.entity_to_archetype: Dict[int, Archetype] = {}
        self.entity_components: Dict[int, _CompDataT] = {}
        self.free_ids: List[int] = []
        self.next_entity_id: int = 0
        self.num_entities: int = 0

        self.systems: List[System] = []

        self.query_cache: Dict[
            int, Tuple[List[Tuple[int, _CompDataT]], int]] = {}
        self.world_version: int = 0

        self.event_bus = EventBus()
        self.component_registry = ComponentRegistry()

    def _invalidate_query_cache(self) -> None:
        """clear cache and update world version"""
        self.query_cache.clear()
        self.world_version += 1

    def _get_archetype(self, signature: int) -> Archetype:
        """Get archetype by signature"""
        if signature not in self.archetypes:
            archetype = Archetype(signature)
            for comp_type in self.component_registry.components:
                bit = self.component_registry.get_bit(comp_type)
                if signature & bit:
                    archetype.storage[comp_type] = []
            self.archetypes[signature] = archetype
        return self.archetypes[signature]

    def register_component(
            self,
            comp_type: Type[Component],
            instance: Optional[Component] = None
    ) -> None:
        """Register a component instance for the given type.

        If instance is None, the world will create one using the default constructor.
        If the component is already registered - the function does nothing.

        Notes:
            The intended use of the World class is to have a single component instance
            per component type.
            Since the function is passed if a component is registered, trying to
            register a new component instance for an existing component type will raise
            an exception.

        Args:
            comp_type (Type[Component]): the type of the component to register
            instance (Component): optional - an initialized component instance. if not
                provided - an instance will be created assuming the default constructor.
                useful for components with a complex or optional constructor.

        Raises:
            ValueError - tried to register a component instance for a type that is
            already registered
        """
        if comp_type not in self.component_registry:
            if instance is None:
                instance = comp_type()
            self.component_registry[comp_type] = instance
        elif instance is not None:
            # trying to manually register a component instance for an existing component
            # raises an error
            raise ValueError(
                f"Cannot register a new instance for component type {comp_type} - "
                f"The component type is already registered for this world.")

    def get_component_instance(self, comp_type: Type[Component]) -> Component:
        """
        Return the registered component for the given type, initializing it if necessary
        """
        if comp_type not in self.component_registry:
            self.register_component(comp_type)
        return self.component_registry[comp_type]

    def register_system(self, system: System) -> None:
        system.initialize(self)
        self.systems.append(system)
        self.systems.sort(key=lambda s: s.priority)

    def create_entity(
        self,
        components: List[Type[Component]],
        component_initial_data: Optional[
            Dict[Type[Component], Union[Tuple[Number, ...], Number]]] = None
    ) -> int:
        """Create an entity with a list of component types.

        Each component type is looked up in the registry (and created automatically
        if needed), and the resulting mapping is stored for that entity. Optionally,
        a mapping of initial data can be supplied.

        Args:
            components (List[Type[Component]]): list of component types to be attached
                to the entity.
            component_initial_data: dictionary of {type: data} with component types and
                their initialization data (tuple that matches the component dimension).
                if not provided, the default defined in the Component class will be
                used.

        Returns:
            the new entity's id
        """
        comp_data: _CompDataT = {
            comp_type: self.get_component_instance(comp_type)
            for comp_type in components
        }
        if self.free_ids:
            entity_id = self.free_ids.pop()
        else:
            entity_id = self.next_entity_id
            self.next_entity_id += 1
        signature = self.component_registry.compute_signature(components)
        archetype = self._get_archetype(signature)
        archetype.add_entity(entity_id, comp_data)
        self.entity_to_archetype[entity_id] = archetype
        self.entity_components[entity_id] = comp_data.copy()
        for comp_type, comp_instance in comp_data.items():
            init_val = None
            if component_initial_data and comp_type in component_initial_data:
                init_val = component_initial_data[comp_type]
            comp_instance.add(entity_id, init_val)
        self._invalidate_query_cache()
        self.num_entities += 1
        return entity_id

    def remove_entity(self, entity_id: int) -> None:
        """Remove an entity from the world.

        This action invalidates the query cache.

        Args:
            entity_id (int): the id of the entity to be removed
        """
        if entity_id not in self.entity_to_archetype:
            warnings.warn("Entity not found.")
            return
        archetype = self.entity_to_archetype.pop(entity_id)
        archetype.remove_entity(entity_id)
        for comp_type in self.entity_components[entity_id]:
            comp = self.get_component_instance(comp_type)
            comp.remove(entity_id)
        self.entity_components.pop(entity_id, None)
        self.free_ids.append(entity_id)
        self._invalidate_query_cache()
        self.num_entities -= 1

    def add_component(
        self,
        entity_id: int,
        comp_type: Type[Component],
        initial_data: Optional[Union[Tuple[Number, ...], Number]] = None
    ) -> None:
        """Add a component to an existing entity.

        The component is added to the entity's mapping and its data is stored via its
        own array.
        This action updates the Archetype of the entity and invalidates the query cache

        Args:
            entity_id (int): the entity to add the component to
            comp_type (Type[Component]): the type of the component to be added
            initial_data: tuple with initial data for the component. must match the
                component dimension (can be a single value for 1d components).
                if not provided - the default from the Component class will be used.
        """
        current = self.entity_components.get(entity_id)
        if current is None:
            raise ValueError("Entity does not exist.")
        if comp_type in current:
            raise ValueError("Entity already has this component.")
        old_archetype = self.entity_to_archetype[entity_id]
        old_archetype.remove_entity(entity_id)
        current[comp_type] = self.get_component_instance(comp_type)
        self.entity_components[entity_id] = current
        new_signature = self.component_registry.compute_signature(list(current.keys()))
        new_archetype = self._get_archetype(new_signature)
        new_archetype.add_entity(entity_id, current)
        self.entity_to_archetype[entity_id] = new_archetype
        current[comp_type].add(entity_id, initial_data)
        self._invalidate_query_cache()

    def remove_component(self, entity_id: int, comp_type: Type[Component]) -> None:
        """Remove a component from an entity.

        This action invalidates the query cache and updates the entity's archetype.

        Args:
            entity_id (int): the entity to add the component to
            comp_type (Type[Component]): the type of the component to be removed
        """
        current = self.entity_components.get(entity_id)
        if current is None:
            raise ValueError("Entity does not exist.")
        if comp_type not in current:
            return
        old_archetype = self.entity_to_archetype[entity_id]
        old_archetype.remove_entity(entity_id)
        current.pop(comp_type)
        self.entity_components[entity_id] = current
        new_signature = self.component_registry.compute_signature(list(current.keys()))
        new_archetype = self._get_archetype(new_signature)
        new_archetype.add_entity(entity_id, current)
        self.entity_to_archetype[entity_id] = new_archetype
        comp_instance = self.get_component_instance(comp_type)
        comp_instance.remove(entity_id)
        self._invalidate_query_cache()

    def query(self, required_comp_types: List[Type[Component]]
              ) -> List[Tuple[int, _CompDataT]]:
        """Query entities that have at least the required component types.

        Returns a list of tuples, each containing an entity id and a dict mapping
        component types to their instances.

        Query results are cached until the world changes (entity removed/modified).

        Args:
            required_comp_types (List[Type[Component]]): list of component. each queried
                entity will have at least all of these components.

        Returns:
            list of tuples:
                id (int): the id of a queried entity
                components data (Dict): dictionary of {type: instance} for all the
                    components of the entity
        """
        query_mask = 0
        for comp_type in required_comp_types:
            query_mask |= self.component_registry.get_bit(comp_type)
        cache_entry = self.query_cache.get(query_mask)
        if cache_entry is not None:
            cached_result, version = cache_entry
            if version == self.world_version:
                return cached_result

        results: List[Tuple[int, _CompDataT]] = []
        for archetype in self.archetypes.values():
            if (archetype.signature & query_mask) == query_mask:
                for idx in range(len(archetype.entities)):
                    entity_id = archetype.entities[idx]
                    entity_data: _CompDataT = {}
                    for comp_type in required_comp_types:
                        entity_data[comp_type] = archetype.storage[comp_type][idx]
                    results.append((entity_id, entity_data))
        self.query_cache[query_mask] = (results, self.world_version)
        return results

    def update_systems(self, dt: float, group: Optional[str] = None) -> None:
        """Update the systems in the world

        Calls the update() method of each registered system.
        If a system is disabled - skip the update.
        Optionally - choose a specific group of systems and only update them

        Args:
            dt (float): time since last system update
            group (Optional[str]): name of the groups to update. only systems with
                `system.group == group` will be updated.
                If None - update all systems.
        """
        for system in self.systems:
            if system.enabled and (group is None or system.group == group):
                system.update(self, dt)

    def update(self, dt: float, group: Optional[str] = None) -> None:
        """Update the world

        Calls the update() method of each registered system and of the event bus.
        If a system is disabled - skip the update.
        Optionally - choose a specific group of systems and only update them

        Args:
            dt (float): time since last system update
            group (Optional[str]): name of the groups to update. only systems with
                `system.group == group` will be updated.
                If None - update all systems.
        """

        self.update_systems(dt, group)
        self.event_bus.update()
