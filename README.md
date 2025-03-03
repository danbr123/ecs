# ECS - Entity Component System for Python

An [Entity Component System](https://en.wikipedia.org/wiki/Entity_component_system) (ECS) framework for python that provides a foundation for games and simulations.
This package supports:

- **Efficient Storage & Querying**:
    
    Entities are grouped by their component composition using bitmask signatures and query caching.

- **High Performance component type**:

    Define your own components by subclassing the base Component class.

    Optionally, use high-performance sparse components that utilizes numpy arrays for batch operations.

- **Dual-Mode Event Bus**:

    Supports both synchronous and asynchronous event dispatch using weak references for subscriber cleanup.


### This package contains:

- Base `Component` class - every component must inherit from this class.
- Base `System` class - provides a basic interface for `System` objects. all systems must inherit from it.
- `SparseComponent` class - This class stores the data of all entities in a numpy array, which in specific cases allows for efficient batch processing.
    
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
from core.sparse.sparse_component import SparseComponent

# Define a simple component.
class Position(Component):
    __slots__ = ("x", "y")
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

# Define a simple system.
class MovementSystem(System):
    def update(self, world: World, dt: float) -> None:
        for entity_id, comps in world.query([Position]):
            pos = comps[Position]
            pos.x += 100 * dt  # Move right at 100 pixels/second

# Create the ECS world.
world = World()

# Create an entity with a Position component.
world.create_entity({Position: Position(0, 0)})

# Register the movement system.
world.register_system(MovementSystem())

# Optionally, subscribe to events.
def on_event(event: Event) -> None:
    print("Event received:", event)

world.event_bus.subscribe(Event, on_event)

# In your update loop:
dt = 1 / 60.0  # Example delta time for 60 FPS.
world.update(dt)  # Updates systems and processes asynchronous events.
```