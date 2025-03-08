from typing import Callable, Dict, List, Optional, Type, TypeVar
from weakref import ref, WeakMethod


class Event:
    """Base class for events."""

    pass


_T = TypeVar("_T", bound=Event)


WeakCallable = Callable[[], Optional[Callable[[_T], None]]]


class EventBus:
    """Stores and dispatches events.

    An event bus that supports both synchronous and asynchronous event dispatch with
    double buffering.

    Synchronous events are dispatched immediately upon publication.
    Asynchronous events are queued and processed in the next update cycle (frame)
    via double buffering, ensuring events published during one frame aren't processed
    until the following frame.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[Type[_T], List[WeakCallable]] = {}
        # Two buffers for asynchronous events.
        self._current_async_queue: List[_T] = []
        self._next_async_queue: List[_T] = []

    def subscribe(self, event_type: Type[_T], handler: Callable[[_T], None]) -> None:
        """Subscribe a handler to a specific event type.

        Whenever an event of that type is dispatched, all subscribers will be called
        with that event. The timing depends on whether the publishing was sync or async.

        Args:
            event_type (Type[Event]): The type of event to subscribe to.
            handler (Callable[[Event], None]): The function to call when the event is
                published.

        Notes:
            The handler is stored as a weak reference. This means that the original
            reference has to be active for it to be called.
            This also means that this feature does not work with `lambda` functions.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        # Callback to remove dead references.
        def _remove(_weak_handler) -> None:
            try:
                self._subscribers[event_type].remove(_weak_handler)
            except ValueError:
                pass

        try:
            weak_handler = WeakMethod(handler, _remove)
        except TypeError:
            weak_handler = ref(handler, _remove)
        self._subscribers[event_type].append(weak_handler)

    def unsubscribe(self, event_type: Type[_T], handler: Callable[[_T], None]) -> None:
        """Unsubscribe a handler from a specific event type.

        Args:
            event_type (Type[Event]): The type of event.
            handler (Callable[[Event], None]): The handler to remove.
        """
        if event_type in self._subscribers:
            for weak_handler in self._subscribers[event_type][:]:
                actual = weak_handler()
                if actual is None or actual == handler:
                    self._subscribers[event_type].remove(weak_handler)

    def publish_sync(self, event: _T) -> None:
        """Publish an event synchronously.

        The event is immediately dispatched to all subscribers registered for its type.

        Args:
            event (Event): The event to publish.
        """
        if not isinstance(event, Event):
            raise TypeError("Published event must be an instance of Event")
        event_type = type(event)
        for weak_handler in self._subscribers.get(event_type, []):
            actual = weak_handler()
            if actual is not None:
                actual(event)

    def publish_async(self, event: _T) -> None:
        """Publish an event asynchronously.

        The event is added to the asynchronous queue and will be processed
        in the next update cycle.

        Args:
            event (Event): The event to publish.
        """
        if not isinstance(event, Event):
            raise TypeError("Published event must be an instance of Event")
        self._next_async_queue.append(event)

    def process_async(self) -> None:
        """
        Process all queued asynchronous events.

        Uses double buffering to ensure events published in the current frame
        aren't processed until the next update cycle.
        """
        # Swap queues and reset next queue.
        self._current_async_queue, self._next_async_queue = self._next_async_queue, []
        for event in self._current_async_queue:
            event_type = type(event)
            for weak_handler in self._subscribers.get(event_type, []):
                actual = weak_handler()
                if actual is not None:
                    actual(event)
        self._current_async_queue.clear()

    def update(self) -> None:
        """Update the event bus by processing asynchronous events."""
        self.process_async()
