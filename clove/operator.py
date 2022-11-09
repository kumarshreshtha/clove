from __future__ import annotations

import dataclasses
from typing import Optional, Sequence
import weakref

from clove import backend as backend_lib
from clove import grad_mode
from clove import variable


def prop_grad(inp):
    return isinstance(inp, variable.Variable) and inp.requires_grad


fn_registry = {}


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
                 variable: variable.Variable = None,
                 backend: backend_lib.Backend = None):

        cls_name = self.__class__.__name__

        class Cache(dict):
            def __getattr__(self, name):
                if name not in self:
                    raise RuntimeError(
                        f"called vjp on {cls_name} after"
                        " releasing the saved tensors. In order to call"
                        " vjp multiple times please set "
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
        self.backend = (
            backend if backend is not None else backend_lib.get_backend())

    def __init_subclass__(
            cls,
            fn_name: Optional[str] = None,
            symbol: Optional[str] = None) -> None:
        cls.symbol = symbol if symbol is not None else cls.__name__
        if fn_name is not None:
            fn_registry[fn_name] = cls
            cls.fn_name = fn_name

    @property
    def next_ops(self):
        return self._children

    @property
    def variable(self):
        return self._var_ref() if self._var_ref is not None else None

    @variable.setter
    def variable(self, value):
        self._var_ref = weakref.ref(value)

    def forward(self, *args, **kwargs) -> variable.Variable:
        raise NotImplementedError(
            f"forward pass for {self.__class__.__name__} has not been"
            " implemented")

    def jvp(self, grad_in: variable.Variable):
        raise NotImplementedError(
            f"jvp pass for {self.__class__.__name__} has not been"
            " implemented")

    def vjp(self, grad_out: variable.Variable):
        raise NotImplementedError(
            f"vjp pass for {self.__class__.__name__} has not been"
            " implemented")

    @classmethod
    def apply(cls, *args, **kwargs):
        children = []
        requires_grad = grad_mode.is_grad_enabled()
        backend = None
        if requires_grad:
            inputs = [*args, *kwargs.values()]
            for arg in inputs:
                if isinstance(arg, variable.Variable):
                    backend = arg.backend if backend is None else backend
                    if backend != arg.backend:
                        raise RuntimeError(
                            "Conflicting backend types in inputs."
                            f" {backend} vs {arg.backend}")
                if prop_grad(arg):
                    op = _LeafOp(arg) if arg.op is None else arg.op
                    children.append(op)
                else:
                    children.append(None)
            requires_grad = any([child is not None for child in children])
        op = cls(children=children if requires_grad else [],
                 requires_grad=requires_grad,
                 backend=backend)
        # ops have their own vjp. therefore the operations they do within
        #  their forward should be detached from the computation graph.
        with grad_mode.set_grad_enabled(False):
            out = op.forward(*args, **kwargs)
        out.requires_grad = requires_grad
        out.op = op if requires_grad else None
        op.variable = out
        return out

    def clear_cache(self):
        self._cache.clear()

    def evaluate(self, *args, **kwargs):
        return variable.Variable(
            self.backend.resolve(self.__class__)(
                *[arg.data if isinstance(arg, variable.Variable) else arg
                  for arg in args],
                **{k: v.data if isinstance(v, variable.Variable) else v
                   for k, v in kwargs.items()}),
            self.backend)


class _LeafOp(Operator, symbol="leaf"):

    def __init__(self, variable: variable.Variable):
        super().__init__(variable=variable,
                         requires_grad=variable.requires_grad)
        self.symbol = (
            variable.name if variable.name is not None else self.symbol)
