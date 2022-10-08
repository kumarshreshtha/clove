from __future__ import annotations

import abc
from typing import Any, Callable, Sequence


class Backend:
    backends = dict()
    _current = None

    def __init_subclass__(cls, name) -> None:
        cls.name = name
        cls.backends[name] = cls

    @abc.abstractclassmethod
    def creation_routines(cls) -> Sequence[Callable]:
        ...

    @abc.abstractclassmethod
    def resolve(cls, op, *args, **kwargs) -> Any:
        ...

    @classmethod
    def current_backend(cls) -> Backend:
        value = cls._current
        if value is None:
            raise RuntimeError("Backend accessed before assignment.")
        return value

    @classmethod
    def set_backend(cls, backend: Backend):
        cls._current = backend


def backends():
    yield from Backend.backends.items()


def backend_from_name(name: str):
    if not name in Backend.backends:
        raise ValueError(f"backend {name} is not registered.")
    return Backend.backends[name]


def has_backend(name: str):
    return name in Backend.backends


def get_backend():
    return Backend.current_backend()


def set_backend(name, /):
    Backend.set_backend(Backend.backends[name])
