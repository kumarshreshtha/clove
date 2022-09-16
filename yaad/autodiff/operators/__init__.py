from .base import _FunctionalFactory, Operator, LeafOp

for __name, __fn in _FunctionalFactory.registry.items():
    globals()[__name] = __fn
