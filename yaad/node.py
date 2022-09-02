"""_summary_"""

from __future__ import annotations

from typing import Optional
import warnings

from yaad import autograd
from yaad import ops


# TODO: add zero grad.


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

    def backward(self, _grad_output=None):
        # TODO: add retain_graph and create graph
        # TODO: maybe changing the requires_grad here is all we need for higher
        # order derivatives.
        autograd.backward(self, _grad_output)

    def __repr__(self):
        data_repr = f"data={self.data}"
        grad_repr = (f", grad={self.grad}" if self.requires_grad
                     and (self.is_leaf() or self.retains_grad)
                     else "")
        return f"Node({data_repr}{grad_repr})"

    def __add__(self, other):
        return ops.add(self, other)

    def __mul__(self, other):
        return ops.multiply(self, other)

    def describe(self):
        # TODO: returns pretty formatted self.name, self.data,
        # self.requires_grad, self.is_leaf, if leaf and non None grad then
        # grad.
        ...
