from abc import ABC


class Component(ABC):
    """Base class for Components

    Every component must inherit from this class in order to be processed properly.
    For frequently used components, it is highly recommended to define `__slots__`
    or wrap the class with `@dataclass(slots=True)`
    """
    __slots__ = ()
    sparse = False
