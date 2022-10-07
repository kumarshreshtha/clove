import abc
from typing import Any, Callable, Mapping, Sequence

from clove import definitions

_backends = dict()


class Backend:
    def __init_subclass__(cls, name) -> None:
        cls.name = name
        _backends[name] = cls

    @abc.abstractclassmethod
    def creation_routines(cls) -> Sequence[Callable]:
        ...

    @abc.abstractclassmethod
    def resolve(cls, defn, binding) -> Any:
        ...


def backends():
    yield from _backends.items()


def backend_from_name(name: str):
    if not name in _backends:
        raise ValueError(f"backend {name} is not registered.")
    return _backends[name]


def has_backend(name: str):
    name in _backends
