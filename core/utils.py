from abc import ABCMeta


class Singleton(ABCMeta):
    """Singleton metaclass

    For every class with this metaclass, only a single instance can be created.
    TODO - handle threading
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
