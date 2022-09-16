from __future__ import annotations

import abc
import functools
import inspect
from typing import Optional, Sequence
import weakref

from yaad import grad_mode
from yaad.autodiff import engine
from yaad.autodiff import node


def prop_grad(inp):
    return isinstance(inp, node.Node) and inp.requires_grad


def wrap_functional(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


class _FunctionalFactory:
    registry = {}

    @classmethod
    def register_op(cls, name, op: Operator):
        new_fn = wrap_functional(op.apply)
        new_fn.__doc__ = op.forward.__doc__
        new_fn.__annotations__ = op.forward.__annotations__
        new_fn.__defaults__ = op.forward.__defaults__
        new_fn.__name__ = name
        sig = inspect.signature(op.forward)
        new_sig = sig.replace(
            parameters=[param for param in sig.parameters.values()
                        if param.name != "self"])
        new_fn.__signature__ = new_sig
        cls.registry[name] = new_fn


class Operator(abc.ABC):

    def __init__(self,
                 children: Sequence[Operator] = (),
                 requires_grad: bool = False,
                 variable: node.Node = None):
        self._cache = {}
        self._children = tuple(children)
        self.requires_grad = requires_grad
        self._var_ref = weakref.ref(variable) if variable is not None else None
        self.grad_store = engine.GradStore()

    def __init_subclass__(cls,
                          fn_name=None,
                          symbol: Optional[str] = None) -> None:
        cls.symbol = symbol if symbol is not None else cls.__name__
        if fn_name is not None:
            _FunctionalFactory.register_op(fn_name, cls)

    @property
    def next_ops(self):
        return self._children

    @property
    def variable(self):
        return self._var_ref() if self._var_ref is not None else None

    @variable.setter
    def variable(self, value):
        self._var_ref = weakref.ref(value)

    @abc.abstractmethod
    def forward(self, *args):
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grad_output: node.Node):
        raise NotImplementedError

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

    def __init__(self, variable):
        super().__init__(variable=variable,
                         requires_grad=variable.requires_grad)
        self.symbol = (
            variable.name if variable.name is not None else self.symbol)

    def forward(self, inp: node.Node):
        return inp

    def backward(self, grad_output: node.Node):
        return grad_output
