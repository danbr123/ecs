from numbers import Number
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

from core.component import Component
from core.system import System
from sparse.sparse_component import SparseComponent


# used to create a bitmask for component composition
_component_bit_registry: Dict[type, int] = {}
_next_component_bit: int = 1


def get_component_bit(component_type: type) -> int:
    """Return a unique bit for the given component type.

    If the component type has not been seen before, assign a new bit.
    """
    global _next_component_bit
    if component_type not in _component_bit_registry:
        _component_bit_registry[component_type] = _next_component_bit
        _next_component_bit <<= 1
    return _component_bit_registry[component_type]


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
        self.storage: Dict[type, List[Component]] = {}
        self.index_map: Dict[int, int] = {}  # entity_id -> index

    def add_entity(self, entity_id: int, components: Dict[type, Component]) -> None:
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

    def remove_entity(self, entity_id: int) -> Optional[Dict[type, Any]]:
        """
        Remove an entity using swap-and-pop.
        Returns the removed component data (for potential reuse) or None if not found.
        """
        if entity_id not in self.index_map:
            return None
        index = self.index_map[entity_id]
        last_index = len(self.entities) - 1
        removed_data: Dict[type, Any] = {}

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
      - Efficient cleanup via free lists and swap-and-pop removal.
    """

    def __init__(self) -> None:
        # Archetype-based storage
        self.archetypes: Dict[int, Archetype] = {}
        self.entity_to_archetype: Dict[int, Archetype] = {}
        self.entity_components: Dict[int, Dict[type, Component]] = {}
        self.free_ids: List[int] = []
        self.next_entity_id: int = 0

        # Registry for sparse components
        # Keyed by component type, value is an instance of SparseComponent
        self.sparse_components: Dict[type, SparseComponent] = {}

        # Systems
        self.systems: List[System] = []

        # Query cache
        self.query_cache: Dict[int, Tuple[List[Dict[type, Any]], int]] = {}
        self.world_version: int = 0

    def _invalidate_query_cache(self) -> None:
        self.query_cache.clear()
        self.world_version += 1

    def _compute_signature(self, components: Dict[type, Any]) -> int:
        signature = 0
        for comp_type in components.keys():
            signature |= get_component_bit(comp_type)
        return signature

    def _get_archetype(self, signature: int) -> Archetype:
        if signature not in self.archetypes:
            archetype = Archetype(signature)
            # Preinitialize storage for each component type present.
            for comp_type in _component_bit_registry:
                bit = get_component_bit(comp_type)
                if signature & bit:
                    archetype.storage[comp_type] = []
            self.archetypes[signature] = archetype
        return self.archetypes[signature]

    def register_sparse_component(self, sparse_component: SparseComponent) -> None:
        """
        Register a SparseComponent instance for a given component type.
        The world will manage its invalidation when entities are removed.
        """
        if (_t := type(sparse_component)) in self.sparse_components:
            raise ValueError(f"Component {_t} cannot be registered twice")
        self.sparse_components[_t] = sparse_component

    def register_system(self, system: System) -> None:
        self.systems.append(system)
        self.systems.sort(key=lambda s: s.priority)

    def create_entity(
            self,
            components: Dict[type, Component],
            sparse_components_data: Dict[type, Union[Tuple[Number, ...], Number]] = None
    ) -> int:
        """
        Create an entity with given component data.
        For sparse components, also update the corresponding SparseComponent.
        """
        if self.free_ids:
            entity_id = self.free_ids.pop()
        else:
            entity_id = self.next_entity_id
            self.next_entity_id += 1
        signature = self._compute_signature(components)
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
        """
        Remove an entity from the world.
        Also invalidate its slot in any registered SparseComponent.
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

    def add_component(self, entity_id: int, comp_type: type, data: Any) -> None:
        """
        Add a component to an existing entity.
        For sparse components, update the corresponding storage.
        """
        current = self.entity_components.get(entity_id)
        if current is None:
            raise ValueError("Entity does not exist.")
        if comp_type in current:
            raise ValueError("Entity already has this component.")
        old_archetype = self.entity_to_archetype[entity_id]
        old_archetype.remove_entity(entity_id)
        current[comp_type] = data
        self.entity_components[entity_id] = current
        new_signature = self._compute_signature(current)
        new_archetype = self._get_archetype(new_signature)
        new_archetype.add_entity(entity_id, current)
        self.entity_to_archetype[entity_id] = new_archetype

        if comp_type in self.sparse_components:
            self.sparse_components[comp_type].add(entity_id, data)
        self._invalidate_query_cache()

    def remove_component(self, entity_id: int, comp_type: type) -> None:
        """
        Remove a component from an entity.
        Also update the corresponding sparse component.
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
        new_signature = self._compute_signature(current)
        new_archetype = self._get_archetype(new_signature)
        new_archetype.add_entity(entity_id, current)
        self.entity_to_archetype[entity_id] = new_archetype

        if comp_type in self.sparse_components:
            self.sparse_components[comp_type].remove(entity_id)
        self._invalidate_query_cache()

    def query(self, required_comp_types: List[type]) -> List[Dict[type, Component]]:
        """
        Query entities that have at least the required component types.
        Returns a list of dicts mapping component types to data.

        Query results are cached until the world changes.
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

        results: List[Dict[type, Component]] = []
        for archetype in self.archetypes.values():
            if (archetype.signature & query_mask) == query_mask:
                # Iterate over entities in this archetype.
                for idx in range(len(archetype.entities)):
                    entity_data: Dict[type, Component] = {}
                    for comp_type in required_comp_types:
                        entity_data[comp_type] = archetype.storage[comp_type][idx]
                    results.append(entity_data)
        self.query_cache[query_mask] = (results, self.world_version)
        return results

    def update_systems(self, dt: float) -> None:
        """
        Update all registered systems.
        Systems typically call query() and modify components.
        """
        for system in self.systems:
            if system.enabled:
                system.update(self, dt)
        # Optionally, perform deferred updates or cleanups here.

    def update(self, dt: float) -> None:
        """
        Main update loop: update systems, and perform world-level maintenance.
        """
        self.update_systems(dt)
