import abc
from typing import Callable, Mapping, Sequence

_registry = dict()


class Backend:
    def __init_subclass__(cls, name) -> None:
        _registry[name] = cls

    @abc.abstractclassmethod
    def creation_ops(cls) -> Sequence[Callable]:
        ...

    @abc.abstractclassmethod
    def fn_associations(cls) -> Mapping[str, Callable]:
        ...


def set_backend(backend, /):
    ...


def get_backend():
    ...


def backends():
    yield from _registry.items()
