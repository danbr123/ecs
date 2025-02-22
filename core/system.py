from abc import ABC, abstractmethod
from typing import Any, Optional


class System(ABC):
    """Abstract base class for systems in the ECS framework."""

    def __init__(
            self,
            priority: float = 10.0,
            enabled: bool = True,
            name: Optional[str] = None
    ) -> None:
        """Initialize a new system

        Args:
            priority: number that dictates the update order of the system. higher number
                means the system will be updated earlier than systems with a lower
                number. can be a float or negative value, only the differences are
                used.
            enabled: flag that checks if the `update()` function should be called
            name: optional name of the system - class name by default
        """
        self.priority = priority
        self.enabled = enabled
        self.name = name or self.__class__.__name__

    def initialize(self, world: Any) -> None:
        """
        Optional hook called when the system is added to the world.
        Use this for one-time setup (resource allocation, caching, etc.).
        """
        pass

    @abstractmethod
    def update(self, world: Any, dt: float) -> None:
        """
        Called every frame/tick if the system is enabled.
        Implement your system logic here.
        """
        pass

    def shutdown(self, world: Any) -> None:
        """
        Optional hook called when the system is removed from the world,
        or when the world is shutting down.
        Use this to clean up resources.
        """
        pass

    def reset(self) -> None:
        """
        Optional method to reset system state, if applicable.
        """
        pass

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False
