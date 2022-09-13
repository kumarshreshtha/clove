
from . import ops

for name, fn in ops.FunctionalFactory._registry.items():
    globals()[name] = fn
