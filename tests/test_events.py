# Tests Generated with ChatGPT
# TODO - improve tests, test edge cases

from ecs.event import Event, EventBus


class DummyEvent(Event):
    def __init__(self, value):
        self.value = value


class DummyHandler:
    def __init__(self):
        self.called = False
        self.event_value = None

    def handle(self, event: DummyEvent) -> None:
        self.called = True
        self.event_value = event.value


def test_publish_sync():
    bus = EventBus()
    handler = DummyHandler()
    bus.subscribe(DummyEvent, handler.handle)
    event = DummyEvent(42)
    bus.publish_sync(event)
    assert handler.called
    assert handler.event_value == 42


def test_unsubscribe():
    bus = EventBus()
    handler = DummyHandler()
    bus.subscribe(DummyEvent, handler.handle)
    bus.unsubscribe(DummyEvent, handler.handle)
    handler.called = False
    bus.publish_sync(DummyEvent(42))
    assert not handler.called


def test_publish_async():
    bus = EventBus()
    handler = DummyHandler()
    bus.subscribe(DummyEvent, handler.handle)
    bus.publish_async(DummyEvent(99))
    assert not handler.called
    bus.update()
    assert handler.called
    assert handler.event_value == 99


def test_double_buffering():
    bus = EventBus()
    handler = DummyHandler()
    bus.subscribe(DummyEvent, handler.handle)
    bus.publish_async(DummyEvent(1))
    bus.publish_async(DummyEvent(2))
    assert not handler.called
    bus.update()
    # The last event processed will be the last one in the first queue.
    assert handler.called
    assert handler.event_value == 2
    handler.called = False
    bus.publish_async(DummyEvent(3))
    bus.update()
    assert handler.called
    assert handler.event_value == 3


def test_lambda_subscription():
    bus = EventBus()
    # Lambdas won't work reliably with weak references.
    bus.subscribe(DummyEvent, lambda e: None)
    # Ensure no error occurs.
    bus.publish_sync(DummyEvent(0))
