
from . import ops
import functools


def warp_call(func, cls):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(cls, *args, **kwargs)
    return wrapper


for name, fn in ops.FunctionalFactory._registry.items():
    globals()[name] = fn
