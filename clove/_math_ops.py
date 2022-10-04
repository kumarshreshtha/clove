from __future__ import annotations

import numbers
from typing import Optional, Union

from clove import operator
from clove import variable
from clove import _registry


def is_number(x):
    return isinstance(x, numbers.Number)


def get_data(array: Union[variable.Variable, numbers.Number]):
    if isinstance(array, variable.Variable):
        return array.data
    return array


class CloneOp(operator.Operator,
              implements=_registry.Function.CLONE,
              symbol="clone"):
    def forward(self, x: variable.Variable):
        return variable.Variable(self.fn(get_data(x)))

    def backward(self, grad_out: variable.Variable):
        return grad_out


class TransposeOp(operator.Operator,
                  implements=_registry.Function.TRANSPOSE,
                  symbol="T"):

    def forward(self, x: variable.Variable):
        return variable.Variable(self.fn(get_data(x)))

    def backward(self, grad_out: variable.Variable):
        return TransposeOp.apply(grad_out)


class AddOp(operator.Operator,
            implements=_registry.Function.ADD,
            symbol="+"):
    def forward(self,
                x1: variable.Variable,
                x2: Union[variable.Variable, numbers.Number]):
        """Adds two nodes or a node and a number."""
        return variable.Variable(self.fn(get_data(x1), get_data(x2)))

    def backward(self, grad_out: Optional[variable.Variable]):
        return grad_out, grad_out


class MulOp(operator.Operator,
            implements=_registry.Function.MULTIPLY,
            symbol="<&times;>"):
    def forward(self,
                x1: variable.Variable,
                x2: variable.Variable):
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return variable.Variable(self.fn(get_data(x1), get_data(x2)))

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = MulOp.apply(x2, grad_out) if x2 is not None else None
        grad_x2 = MulOp.apply(x1, grad_out) if x1 is not None else None
        return grad_x1, grad_x2


class MatmulOp(operator.Operator,
               implements=_registry.Function.MATMUL,
               symbol="@"):
    def forward(self,
                x1: variable.Variable,
                x2: variable.Variable):
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return variable.Variable(self.fn(get_data(x1), get_data(x2)))

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = (MatmulOp.apply(grad_out, TransposeOp.apply(x2))
                   if x2 is not None else None)
        grad_x2 = (MatmulOp.apply(TransposeOp.apply(x1), grad_out)
                   if x1 is not None else None)
        return grad_x1, grad_x2


class NegOp(operator.Operator,
            implements=_registry.Function.NEGATE,
            symbol="-1*"):
    def forward(self, x):
        return variable.Variable(self.fn(get_data(x)))

    def backward(self, grad_out):
        return NegOp.apply(grad_out)


class MinusOp(operator.Operator,
              implements=_registry.Function.SUBTRACT,
              symbol="-"):
    def forward(self, x1, x2):
        return variable.Variable(self.fn(get_data(x1), get_data(x2)))

    def backward(self, grad_out: variable.Variable):
        return grad_out, NegOp.apply(grad_out)


class ExpOp(operator.Operator,
            implements=_registry.Function.EXP,
            symbol="exp"):
    def forward(self, x: variable.Variable):
        out = variable.Variable(self.fn(get_data(x)))
        self._cache.out = out if self.requires_grad else None
        return out

    def backward(self, grad_out):
        out = self._cache.out
        return MulOp.apply(out, grad_out) if out is not None else None


class LogOp(operator.Operator,
            implements=_registry.Function.LOG,
            symbol="ln"):
    def forward(self, x: variable.Variable):
        self._cache.x = x if self.requires_grad else None
        out = variable.Variable(self.fn(get_data(x)))
        return out

    def backward(self, grad_out: variable.Variable):
        x = self._cache.x
        return (MulOp.apply(PowOp.apply(x, -1), grad_out)
                if x is not None else None)


class PowOp(operator.Operator,
            implements=_registry.Function.POW,
            symbol="**"):
    def forward(self, x1: variable.Variable, x2: variable.Variable):
        out = variable.Variable(self.fn(get_data(x1), get_data(x2)))
        self._cache.x1 = (
            x1 if operator.prop_grad(x1) or operator.prop_grad(x2) else None)
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.out = out if operator.prop_grad(x2) else None
        return out

    def backward(self, grad_out: variable.Variable):
        x1, x2, out = self._cache.x1, self._cache.x2, self._cache.out
        x1_grad = x2_grad = None
        if x2 is not None:
            x1_grad = (
                MulOp.apply(
                    MulOp.apply(x2, PowOp.apply(
                        x1, MinusOp.apply(x2, 1))), grad_out))
        if out is not None:
            x2_grad = MulOp.apply(out, LogOp.apply(x1))
        return x1_grad, x2_grad


class SigmoidOp(operator.Operator,
                implements=_registry.Function.SIGMOID,
                symbol="<&sigma;>"):
    def forward(self, x: variable.Variable):
        out = variable.Variable(self.fn(get_data(x)))
        self._cache.out = out if operator.prop_grad(x) else None
        return out

    def backward(self, grad_out: variable.Variable):
        out = self._cache.out
        return (MulOp.apply(MulOp.apply(out, MinusOp.apply(1, out)), grad_out)
                if out is not None else None)


class TanhOp(operator.Operator,
             implements=_registry.Function.TANH,
             symbol="tanh"):
    def forward(self, x: variable.Variable):
        out = variable.Variable(self.fn(get_data(x)))
        self._cache.out = out if operator.prop_grad(x) else None
        return out

    def backward(self, grad_out: variable.Variable):
        out = self._cache.out
        return (MulOp.apply(MinusOp(1, PowOp.apply(out, 2), grad_out))
                if out is not None else None)
