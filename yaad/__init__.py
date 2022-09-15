
from . import ops
import functools
from .node import Node


def warp_call(func, cls):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(cls, *args, **kwargs)
    return wrapper


for name, fn in ops.FunctionalFactory._registry.items():
    globals()[name] = fn

for name, fn in ops.FunctionalFactory._node_registry.items():
    setattr(Node, name, fn)
