from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


class Resources(MutableMapping[str, Any]):
    """
    Thin wrapper around a dict for ECS 'global state' (render view, camera,
    event queues, input snapshot, etc.). Provides:
      - Stable API boundary (swap internals later without touching systems).
      - Optional typed getters with runtime validation.
      - Namespacing via dot-keys ("render.view", "audio.volume").
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}

    def get(self, key: str, default: Optional[str] = None) -> Any:
        return self._data.get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def set_if_missing(self, key: str, value: Any) -> Any:
        """Set value if one doesn't exist"""
        return self._data.setdefault(key, value)

    def namespace(self, prefix: str) -> "ResourceView":
        """Create a namespaced view of the resources

        Accessing a key 'key' in the namespace will access 'prefix.key' in
        the original resource data.

        Examples:
            >>> r = Resources()
            >>> namespace1 = r.namespace("ns1")
            >>> namespace2 = r.namespace("ns2")
            >>> namespace1["fps"] = 60
            >>> namespace2["fps"] = 30
            >>> namespace1["fps"]  # 60
            >>> r["ns2.fps"]  # 30
        """
        if prefix.endswith("."):
            prefix = prefix[:-1]
        return ResourceView(self, prefix)


@dataclass(frozen=True)
class ResourceView:
    """A namespaced view over Resources: keys are stored as 'prefix.key'"""
    _root: Resources
    _prefix: str

    def _k(self, key: str) -> str:
        if not self._prefix:
            return key
        return f"{self._prefix}.{key}"

    def __getitem__(self, key: str) -> Any:
        return self._root[self._k(key)]

    def __setitem__(self, key: str, value: Any) -> None:
        self._root[self._k(key)] = value

    def __delitem__(self, key: str) -> None:
        del self._root[self._k(key)]

    def get(self, key: str, default: Any = None) -> Any:
        return self._root.get(self._k(key), default)

    def set_if_missing(self, key: str, value: Any) -> Any:
        return self._root.set_if_missing(self._k(key), value)
