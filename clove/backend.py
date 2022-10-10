from __future__ import annotations

import abc


class Backend:
    backends = dict()
    _current = None

    def __init_subclass__(cls, name) -> None:
        cls.name = name
        cls.backends[name] = cls

    @abc.abstractclassmethod
    def resolve(cls, op, *args, **kwargs):
        ...

    @abc.abstractclassmethod
    def resolve_shape(cls, data):
        ...

    # TODO: define your own dtypes will be easier for comparision.
    @abc.abstractclassmethod
    def resolve_dtype(cls, data):
        ...

    @abc.abstractclassmethod
    def resolve_numel(cls, data):
        ...

    @abc.abstractclassmethod
    def make_from_number(cls, data):
        ...

    # TODO: add this when ready for gpu support
    # @abc.abstractclassmethod
    # def resolve_device(cls, data) -> Any:
    #     ...

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


def has_backend(name: str):
    return name in Backend.backends


def get_backend():
    return Backend.current_backend()


def set_backend(name, /):
    if not name in Backend.backends:
        raise ValueError(f"backend {name} is not registered.")
    Backend.set_backend(Backend.backends[name])
