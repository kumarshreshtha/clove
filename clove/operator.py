from __future__ import annotations
import dataclasses

import inspect
from typing import Optional, Sequence
import weakref

from clove import _backend
from clove import grad_mode
from clove import variable


def prop_grad(inp):
    return isinstance(inp, variable.Variable) and inp.requires_grad


class Operator:
    @dataclasses.dataclass
    class GradStore:
        value: Optional[variable.Variable] = None

        def reset(self):
            self.value = None

        def update(self, grad: variable.Variable):
            self.value = self.value + grad if self.value is not None else grad

    def __init__(self,
                 children: Sequence[Operator] = (),
                 requires_grad: bool = False,
                 variable: variable.Variable = None):

        cls_name = self.__class__.__name__

        class Cache(dict):
            def __getattr__(self, name):
                if name not in self:
                    raise RuntimeError(
                        f"called backward on {cls_name} after"
                        " releasing the saved tensors. In order to call"
                        " backward multiple times please set "
                        "`retain_graph=True`.")
                return self[name]

            def __setattr__(self, name, value):
                if not requires_grad:
                    return
                self[name] = value

        self._children = tuple(children)
        self.requires_grad = requires_grad
        self._var_ref = weakref.ref(variable) if variable is not None else None
        self._cache = Cache()
        self.grad_store = self.GradStore()

    def __init_subclass__(
            cls,
            symbol: Optional[str] = None) -> None:
        cls.symbol = symbol if symbol is not None else cls.__name__
        # cls.defn = implements
        # if implements is not None:
        #     definitions.register_operator(implements, cls)

    @property
    def next_ops(self):
        return self._children

    @property
    def variable(self):
        return self._var_ref() if self._var_ref is not None else None

    @variable.setter
    def variable(self, value):
        self._var_ref = weakref.ref(value)

    def forward(self, *args):
        raise NotImplementedError(
            f"forward pass for {self.__class__.__name__} has not been"
            " implemented")

    def backward(self, grad_out: variable.Variable):
        raise NotImplementedError(
            f"backward pass for {self.__class__.__name__} has not been"
            " implemented")

    @classmethod
    def apply(cls, *args):
        children = []
        requires_grad = grad_mode.is_grad_enabled()
        if requires_grad:
            for arg in args:
                if prop_grad(arg):
                    op = _LeafOp(arg) if arg.op is None else arg.op
                    children.append(op)
                else:
                    children.append(None)
            requires_grad = any([child is not None for child in children])
        op = cls(children=children if requires_grad else [],
                 requires_grad=requires_grad)
        # ops have their own backward. therefore the operations they do within
        #  their forward should be detached from the computation graph.
        with grad_mode.set_grad_enabled(False):
            out = op.forward(*args)
        out.requires_grad = requires_grad
        out.op = op if requires_grad else None
        op.variable = out
        return out

    def clear_cache(self):
        self._cache.clear()

    def evaluate(self, *args, **kwargs):
        value = _backend.get_backend().resolve(self, *args, **kwargs)
        return variable.Variable(value)

    @classmethod
    def get_signature(cls):
        sig = inspect.signature(cls.forward)
        return sig.replace(
            parameters=[param for param in sig.parameters.values()
                        if param.name != "self"])


class _LeafOp(Operator, symbol="leaf"):

    def __init__(self, variable: variable.Variable):
        super().__init__(variable=variable,
                         requires_grad=variable.requires_grad)
        self.symbol = (
            variable.name if variable.name is not None else self.symbol)
