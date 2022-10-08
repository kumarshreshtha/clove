from .base import backends, Backend, backend_from_name, has_backend
from . import numpy
from . import cupy

__BACKEND = numpy.Numpy  # defaults to numpy.


def set_backend(backend: str, /):
    global __BACKEND
    __BACKEND = backend_from_name(backend)


def get_backend():
    return __BACKEND
