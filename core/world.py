from numbers import Number
from typing import Dict, List, Optional, Tuple, Union, TypeVar, Type
import warnings

from core.component import Component
from core.event import EventBus
from core.system import System
from core.sparse.sparse_component import SparseComponent


_T = TypeVar("_T", bound=Component)
_CompDataT = Dict[Type[Component], _T]

# used to create a bitmask for component composition
_component_bit_registry: Dict[Type[Component], int] = {}
_next_component_bit: int = 1


def get_component_bit(component_type: Type[Component]) -> int:
    """Return a unique bit for the given component type.

    If the component type has not been seen before, assign a new bit.

    Args:
        component_type(type): Type of the component
    Returns:
        int - The bit of the component type, represented as an integer
    """
    global _next_component_bit
    if component_type not in _component_bit_registry:
        _component_bit_registry[component_type] = _next_component_bit
        _next_component_bit <<= 1
    return _component_bit_registry[component_type]


def _compute_signature(components: _CompDataT) -> int:
    """Get unique signature for a composition of components

    Args:
        components: dictionary of component data - {type: instance}

    Returns:
        an integer that represents the signature of this component composition. each
            component affects a unique bit in that signature.

    """
    signature = 0
    for comp_type in components.keys():
        signature |= get_component_bit(comp_type)
    return signature


class Archetype:
    """An archetype groups entities that share the same component composition.

    It stores:
      - signature: an integer bitmask representing the set of component types.
      - entities: a dense list of entity IDs.
      - storage: a dictionary mapping component types to a dense list of component data.
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
        """
        index = len(self.entities)
        self.entities.append(entity_id)
        self.index_map[entity_id] = index
        # For each component type in the archetype, append the data.
        for comp_type in components:
            if comp_type not in self.storage:
                self.storage[comp_type] = []
            self.storage[comp_type].append(components[comp_type])

    def remove_entity(self, entity_id: int) -> Optional[_CompDataT]:
        """
        Remove an entity using swap-and-pop.
        Returns the removed component data (for potential reuse) or None if not found.
        """
        if entity_id not in self.index_map:
            return None
        index = self.index_map[entity_id]
        last_index = len(self.entities) - 1
        removed_data: _CompDataT = {}

        # Save removed data from each component's storage.
        for comp_type, data_list in self.storage.items():
            removed_data[comp_type] = data_list[index]

        # If not removing the last entity, swap the last entity's data
        # into the removed slot.
        if index != last_index:
            last_entity = self.entities[last_index]
            self.entities[index] = last_entity
            self.index_map[last_entity] = index
            for comp_type, data_list in self.storage.items():
                data_list[index] = data_list[last_index]
        # Remove the last entry.
        self.entities.pop()
        for comp_type, data_list in self.storage.items():
            data_list.pop()
        del self.index_map[entity_id]
        return removed_data


class World:
    """
    The ECS World manages entities, components, archetypes, and systems.

    It supports:
      - Archetype-based storage for regular components (using bit masks).
      - Registration of SparseComponent instances for high-performance components.
      - A system update loop.
      - Query caching to speed up repeated queries.
      - Efficient cleanup via free lists and swap-and‐pop removal.
    """

    def __init__(self) -> None:
        # Archetype-based storage
        self.archetypes: Dict[int, Archetype] = {}
        self.entity_to_archetype: Dict[int, Archetype] = {}
        self.entity_components: Dict[int, _CompDataT] = {}
        self.free_ids: List[int] = []
        self.next_entity_id: int = 0

        # Registry for sparse components
        # Keyed by component type, value is an instance of SparseComponent
        self.sparse_components: Dict[Type[Component], SparseComponent] = {}

        # Systems
        self.systems: List[System] = []

        # Query cache
        self.query_cache: Dict[
            int, Tuple[List[Tuple[int, _CompDataT]], int]] = {}
        self.world_version: int = 0

        # Event bus
        self.event_bus = EventBus()

    def _invalidate_query_cache(self) -> None:
        self.query_cache.clear()
        self.world_version += 1

    def _get_archetype(self, signature: int) -> Archetype:
        if signature not in self.archetypes:
            archetype = Archetype(signature)
            # Pre-initialize storage for each component type present.
            for comp_type in _component_bit_registry:
                bit = get_component_bit(comp_type)
                if signature & bit:
                    archetype.storage[comp_type] = []
            self.archetypes[signature] = archetype
        return self.archetypes[signature]

    def register_sparse_component(self, sparse_component: SparseComponent) -> None:
        """Register a SparseComponent instance.

        The world will manage its invalidation when entities are removed.
        Only a single sparse_component can be created per
        """
        if (_t := type(sparse_component)) in self.sparse_components:
            raise ValueError(f"Component {_t} cannot be registered twice")
        self.sparse_components[_t] = sparse_component

    def register_system(self, system: System) -> None:
        """Add a new system to the world

        Every registered system will be updated when the world is updated unless
        disabled.

        Args:
            system: initialized system
        """
        system.initialize(self)
        self.systems.append(system)
        self.systems.sort(key=lambda s: s.priority)

    def create_entity(
            self,
            components: _CompDataT,
            sparse_components_data: Optional[
                Dict[Type[Component], Union[Tuple[Number, ...], Number]]] = None
    ) -> int:
        """Create an entity with given component data.

        For sparse components, also update the corresponding SparseComponent.

        Args:
            components (Dict[type, Component]): dictionary of component types and
                initialized components that the entity will use.
            sparse_components_data: optional data for sparse components, if not
                specified, the default value from the sparse component class will
                be used.
        """
        if self.free_ids:
            entity_id = self.free_ids.pop()
        else:
            entity_id = self.next_entity_id
            self.next_entity_id += 1
        signature = _compute_signature(components)
        archetype = self._get_archetype(signature)
        archetype.add_entity(entity_id, components)
        self.entity_to_archetype[entity_id] = archetype
        self.entity_components[entity_id] = components.copy()

        for comp_type, sparse in self.sparse_components.items():
            if comp_type in components:
                if (sparse_components_data is None
                        or comp_type not in sparse_components_data):
                    raise ValueError(f"Data from {comp_type} was not provided")
                sparse.add(entity_id, sparse_components_data[comp_type])
        self._invalidate_query_cache()
        return entity_id

    def remove_entity(self, entity_id: int) -> None:
        """Remove an entity from the world.

        Also invalidate its slot in any registered SparseComponent.
        This action invalidates the query cache.

        Args:
            entity_id: entity to remove
        """
        if entity_id not in self.entity_to_archetype:
            warnings.warn("Entity not found.")
            return
        archetype = self.entity_to_archetype.pop(entity_id)
        archetype.remove_entity(entity_id)
        self.entity_components.pop(entity_id, None)

        for sparse in self.sparse_components.values():
            sparse.remove(entity_id)
        self.free_ids.append(entity_id)
        self._invalidate_query_cache()

    def add_component(
            self,
            entity_id: int,
            component: Component,
            sparse_data: Optional[Union[Tuple[Number, ...], Number]] = None
    ) -> None:
        """Add a component to an existing entity.

        Adds the component to the entity components and changes the archetype
        of the entity.
        For sparse components, update the corresponding storage.

        Args:
            entity_id (int): entity to add the component to
            component (Component): component to attach to the entity
            sparse_data (Optional[Union[Tuple[Number, ...], Number]]): initial data for
                sparse components. if None - the default will be used from the sparse
                component class.
        """
        current = self.entity_components.get(entity_id)
        if current is None:
            raise ValueError("Entity does not exist.")
        if (_t := type(component)) in current:
            raise ValueError("Entity already has this component.")
        old_archetype = self.entity_to_archetype[entity_id]
        old_archetype.remove_entity(entity_id)
        current[_t] = component
        self.entity_components[entity_id] = current
        new_signature = _compute_signature(current)
        new_archetype = self._get_archetype(new_signature)
        new_archetype.add_entity(entity_id, current)
        self.entity_to_archetype[entity_id] = new_archetype

        if _t in self.sparse_components:
            self.sparse_components[_t].add(entity_id, sparse_data)
        self._invalidate_query_cache()

    def remove_component(self, entity_id: int, comp_type: _T) -> None:
        """Remove a component from an entity.

        Also update the corresponding sparse component.
        This action invalidates the query cache and updates the Archetype of the entity.

        Args:
            entity_id (int): entity to remove the component from
            comp_type (type): the component type to remove
        """
        current = self.entity_components.get(entity_id)
        if current is None:
            raise ValueError("Entity does not exist.")
        if comp_type not in current:
            raise ValueError("Entity does not have that component.")
        old_archetype = self.entity_to_archetype[entity_id]
        old_archetype.remove_entity(entity_id)
        del current[comp_type]
        self.entity_components[entity_id] = current
        new_signature = _compute_signature(current)
        new_archetype = self._get_archetype(new_signature)
        new_archetype.add_entity(entity_id, current)
        self.entity_to_archetype[entity_id] = new_archetype

        if comp_type in self.sparse_components:
            self.sparse_components[comp_type].remove(entity_id)
        self._invalidate_query_cache()

    def query(self, required_comp_types: List[_T]
              ) -> List[Tuple[int, _CompDataT]]:
        """
        Query entities that have at least the required component types.
        Returns a list of tuples, each containing an entity id and a dict mapping
        component types to data.

        Query results are cached until the world changes (entity removed/modified)

        Args:
            required_comp_types (List) - list of component types queried entities must
                have.

        Returns:
            list of tuples, each representing a queried entity:
                int - id of an entity that matches the query
                Dict[type, Component] - dictionary of components of that entity
        """
        query_mask = 0
        for comp_type in required_comp_types:
            query_mask |= get_component_bit(comp_type)
        # Check cache.
        cache_entry = self.query_cache.get(query_mask)
        if cache_entry is not None:
            cached_result, version = cache_entry
            if version == self.world_version:
                return cached_result

        results: List[Tuple[int, _CompDataT]] = []
        for archetype in self.archetypes.values():
            if (archetype.signature & query_mask) == query_mask:
                # Iterate over entities in this archetype.
                for idx in range(len(archetype.entities)):
                    entity_id = archetype.entities[idx]
                    entity_data: _CompDataT = {}
                    for comp_type in required_comp_types:
                        entity_data[comp_type] = archetype.storage[comp_type][idx]
                    results.append((entity_id, entity_data))
        self.query_cache[query_mask] = (results, self.world_version)
        return results

    def update_systems(self, dt: float) -> None:
        """Update all registered systems.

        Systems typically call query() and modify components.
        Disabled systems are not updated (use System.disable(), System.enable() to
        modify this flag)
        Systems are updated in an order dictated by their priority, with higher priority
        values being updated first.

        Args:
            dt (float): time since last update - used to make update time dependent
                instead of frame-rate dependent
        """
        for system in self.systems:
            if system.enabled:
                system.update(self, dt)

    def update(self, dt: float) -> None:
        """Main update loop: update systems, and perform world-level maintenance.

        Args:
            dt (float): time since last update - used to make update time dependent
                instead of frame-rate dependent
        """
        self.update_systems(dt)
        self.event_bus.update()  # process async events
