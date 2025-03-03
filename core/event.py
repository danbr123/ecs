from typing import Callable, Dict, List, Type


class Event:
    """Base class for events."""
    pass


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
        # Maps event types (subclasses of Event) to lists of handler functions.
        self._subscribers: Dict[Type[Event], List[Callable[[Event], None]]] = {}
        # Two buffers for asynchronous events.
        self._current_async_queue: List[Event] = []
        self._next_async_queue: List[Event] = []

    def subscribe(
            self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """Subscribe a handler to a specific event type.

        Whenever an event of that type is dispatched, all subscribers will be called
        with that event.
        The timing depends on whether the publishing was sync or async.

        Args:
            event_type (Type[Event]): The type of event to subscribe to.
            handler (Callable[[Event], None]): The function to call when the event is published.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)

    def unsubscribe(
            self, event_type: Type[Event], handler: Callable[[Event], None]) -> None:
        """Unsubscribe a handler from a specific event type.

        Args:
            event_type (Type[Event]): The type of event.
            handler (Callable[[Event], None]): The handler to remove.
        """
        if event_type in self._subscribers:
            self._subscribers[event_type].remove(handler)

    def publish_sync(self, event: Event) -> None:
        """Publish an event synchronously.

        The event is immediately dispatched to all subscribers registered for its type.
        Args:
            event (Event): The event to publish.
        """
        if not isinstance(event, Event):
            raise TypeError("Published event must be an instance of Event")
        event_type = type(event)
        for handler in self._subscribers.get(event_type, []):
            handler(event)

    def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously.

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
        # swap queues and reset next queue
        self._current_async_queue, self._next_async_queue = self._next_async_queue, []
        for event in self._current_async_queue:
            event_type = type(event)
            for handler in self._subscribers.get(event_type, []):
                handler(event)
        self._current_async_queue.clear()

    def update(self) -> None:
        """Update the event bus by processing asynchronous events.

        This should be called once per frame (or update cycle).
        """
        self.process_async()
