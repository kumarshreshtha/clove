from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, Optional, Sequence
import weakref


from yaad import grad_mode

from yaad import variable


def prop_grad(inp):
    return isinstance(inp, variable.Variable) and inp.requires_grad


class Operator:

    class GradStore:

        def __init__(self, value: Optional[variable.Variable]):
            self.value = value

        def reset(self):
            self.value = None

        def update(self, grad: variable.Variable):
            # Note: if incoming grad has requires_grad=True then this value becomes
            # part of the graph thus making higher order derivatives possible.
            self.value = self.value + grad if self.value is not None else grad

    def __init__(self,
                 children: Sequence[Operator] = (),
                 requires_grad: bool = False,
                 variable: variable.Variable = None):
        self._cache = {}
        self._children = tuple(children)
        self.requires_grad = requires_grad
        self._var_ref = weakref.ref(variable) if variable is not None else None
        self.grad_store = Operator.GradStore()

    def __init_subclass__(cls,
                          fn_name=None,
                          symbol: Optional[str] = None) -> None:
        cls.symbol = symbol if symbol is not None else cls.__name__
        # if fn_name is not None:
        #     _OperatorsFactory.register_op(fn_name, cls)

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
        raise NotImplementedError()

    def backward(self, grad_output: variable.Variable):
        raise NotImplementedError()

    @classmethod
    def apply(cls, *args):
        children = []
        requires_grad = grad_mode.is_grad_enabled()
        if requires_grad:
            for arg in args:
                if prop_grad(arg):
                    op = LeafOp(arg) if arg.op is None else arg.op
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

    def save_for_backward(self, name, value):
        self._cache[name] = value

    def saved_value(self, name):
        if name not in self._cache:
            raise RuntimeError(f"called backward on {self.__class__.__name__} "
                               "after releasing the saved tensors. In order "
                               "to call backward multiple times please set "
                               "`retain_graph=True`.")
        return self._cache[name]

    # def __repr__(self):
    #     op_repr = f"Operator({self.__class__.__name__}[{self.symbol}])"
    #     var = self.variable
    #     var_repr = if var is not None


class LeafOp(Operator, symbol="leaf"):

    def __init__(self, variable: variable.Variable):
        super().__init__(variable=variable,
                         requires_grad=variable.requires_grad)
        self.symbol = (
            variable.name if variable.name is not None else self.symbol)
