import abc
from typing import Callable, Mapping, Sequence

_registry = dict()


class Backend:
    def __init_subclass__(cls, name) -> None:
        cls.name = name
        _registry[name] = cls

    @abc.abstractclassmethod
    def creation_ops(cls) -> Sequence[Callable]:
        ...

    @abc.abstractclassmethod
    def fn_associations(cls) -> Mapping[str, Callable]:
        ...


def backends():
    yield from _registry.items()


def backend_from_name(name: str):
    if not name in _registry:
        raise ValueError(f"backend {name} is not registered.")
    return _registry[name]
