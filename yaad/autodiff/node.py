"""_summary_"""

from __future__ import annotations

import inspect
from typing import Optional
import warnings

from yaad.autodiff import engine
from yaad.autodiff.operators import ops


class Node:
    """Scalar Node."""

    def __init__(self,
                 data,
                 requires_grad=False,
                 name: str = None):
        self._data = data
        self._grad = None
        self.requires_grad = requires_grad
        self.name = name
        self.op: Optional[ops.Operator] = None
        self._retains_grad = False

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self.is_leaf() and not self.requires_grad:
            self._data = value
        else:
            raise RuntimeError("cannot mutate variable attached to a graph"
                               " in place.")

    @property
    def grad(self):
        if not self.is_leaf() and not self.retains_grad:
            warnings.warn(
                "gradient attribute of non-leaf nodes are not"
                " stored unless `retain_grad` has been explicitly set.")
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def retains_grad(self):
        return self._retains_grad

    def retain_grad(self):
        if self.is_leaf():
            return
        self._retains_grad = True

    def is_leaf(self):
        return self.op is None

    def backward(self, grad_output=None, retain_graph=False):
        engine.backward(self, grad_output, retain_graph)

    def zero_grad(self, set_to_none: bool = False):
        self._grad = None if set_to_none else 0.

    def __neg__(self):
        return -1 * self

    def describe(self):
        # TODO: returns pretty formatted self.name, self.data,
        # self.requires_grad, self.is_leaf, if leaf and non None grad then
        # grad.
        ...

    def __repr__(self):
        grad_repr = f", requires_grad=True" if self.requires_grad else ""
        return f"Node({self.data}{grad_repr})"


def _register_node_methods():
    _OP_REGISTY = dict(
        __add__=ops.AddOp,
        __radd__=ops.AddOp,
        __mul__=ops.MulOp,
        __rmul__=ops.MulOp,
        __sub__=ops.MinusOp,
        __rsub__=ops.MinusOp,
        __pow__=ops.PowOp,
        sigmoid=ops.SigmoidOp
    )
    for name, op in Node._OP_REGISTY.items():
        new_fn = ops.wrap_functional(op.apply)
        new_fn.__doc__ = op.forward.__doc__
        new_fn.__annotations__ = op.forward.__annotations__
        new_fn.__defaults__ = op.forward.__defaults__
        new_fn.__name__ = name
        sig = inspect.signature(op.forward)
        new_sig = sig.replace(
            parameters=[param for param in sig.parameters.values()
                        if param.name != "inp"])
        new_fn.__signature__ = new_sig
        setattr(Node, name, new_fn)


_register_node_methods()
