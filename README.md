# ECS - Entity Component System for Python

An [Entity Component System](https://en.wikipedia.org/wiki/Entity_component_system) (ECS) framework for python that provides a foundation for games and simulations.
This package supports:

- **Efficient Storage & Querying**:
    
    Entities are grouped by their component composition using bitmask signatures and query caching.

- **Optimized Components**:

    Components work with an underlying numpy array, making it easy to perform batch operations, as well as reducing the chance for cache misses.

- **Dual-Mode Event Bus**:

    Supports both synchronous and asynchronous event dispatch using weak references for subscriber cleanup.


### This package contains:

- Base `Component` class - every component must inherit from this class.
- Base `System` class - provides a basic interface for `System` objects. all systems must inherit from it.
    
    The Component stores an array of all entities, regardless of whether they use it. therefore it is recommended to use it for common components that can benefit from vectorized operations.
- `World` class - stores all the entities, systems and components, and provides an interface for inserting new entities, linking components, registring systems
    and querying entities by components.
- `EventBus` class - manages sync and async events - used to subscribe to events and publish events. accessible via the World class using `world.event_bus` but can be used separately.


### Example usage:
```python
from core.world import World
from core.component import Component
from core.system import System
from core.event import Event

```