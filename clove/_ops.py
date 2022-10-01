from __future__ import annotations

import numbers
import numpy as np
from typing import Optional, Union

from clove import operator
from clove import variable
from clove import _registry


def is_number(x):
    return isinstance(x, numbers.Number)


def get_data(array: Union[variable.Variable, np.ndarray, numbers.Number]):
    if isinstance(array, variable.Variable):
        return array.data
    return array

# TODO: update function signatures automatically from backend docs.


class CloneOp(operator.Operator,
              implements=_registry.FunctionNames.CLONE,
              symbol="clone"):
    def forward(self, x: variable.Variable):
        return variable.Variable(x.data.copy())

    def backward(self, grad_output: variable.Variable):
        return grad_output


class TransposeOp(operator.Operator,
                  implements=_registry.FunctionNames.TRANSPOSE,
                  symbol="T"):

    def forward(self, x: variable.Variable):
        return variable.Variable(get_data(x).T)

    def backward(self, grad_output: variable.Variable):
        return TransposeOp.apply(grad_output)


class AddOp(operator.Operator,
            implements=_registry.FunctionNames.ADD,
            symbol="+"):
    def forward(self,
                x1: variable.Variable,
                x2: Union[variable.Variable, numbers.Number]):
        """Adds two nodes or a node and a number."""
        return variable.Variable(np.add(get_data(x1), get_data(x2)))

    def backward(self, grad_output: Optional[variable.Variable]):
        return grad_output, grad_output


class MulOp(operator.Operator,
            implements=_registry.FunctionNames.MULTIPLY,
            symbol="<&times;>"):
    def forward(self,
                x1: variable.Variable,
                x2: variable.Variable):
        self.save_for_backward("x2", x2 if operator.prop_grad(x1) else None)
        self.save_for_backward("x1", x1 if operator.prop_grad(x2) else None)
        return variable.Variable(np.multiply(get_data(x1), get_data(x2)))

    def backward(self, grad_output: variable.Variable):
        x2 = self.saved_value("x2")
        x1 = self.saved_value("x1")
        grad_x1 = MulOp.apply(x2, grad_output) if x2 is not None else None
        grad_x2 = MulOp.apply(x1, grad_output) if x1 is not None else None
        return grad_x1, grad_x2


class MatmulOp(operator.Operator,
               implements=_registry.FunctionNames.MATMUL,
               symbol="@"):
    def forward(self,
                x1: variable.Variable,
                x2: variable.Variable):
        self.save_for_backward(
            "x2", x2 if operator.prop_grad(x1) else None)
        self.save_for_backward(
            "x1", x1 if operator.prop_grad(x2) else None)
        return variable.Variable(np.matmul(get_data(x1), get_data(x2)))

    def backward(self, grad_output: variable.Variable):
        x2 = self.saved_value("x2")
        x1 = self.saved_value("x")
        grad_x1 = (MatmulOp.apply(grad_output, TransposeOp.apply(x2))
                   if x2 is not None else None)
        grad_x2 = (MatmulOp.apply(TransposeOp.apply(x1), grad_output)
                   if x1 is not None else None)
        return grad_x1, grad_x2


class NegOp(operator.Operator,
            implements=_registry.FunctionNames.NEGATE,
            symbol="-1*"):
    def forward(self, x):
        return variable.Variable(np.negative(get_data(x)))

    def backward(self, grad_output):
        return NegOp.apply(grad_output)


class MinusOp(operator.Operator,
              implements=_registry.FunctionNames.SUBTRACT,
              symbol="-"):
    def forward(self, x1, x2):
        return variable.Variable(np.subtract(get_data(x1), get_data(x2)))

    def backward(self, grad_output: variable.Variable):
        return grad_output, NegOp.apply(grad_output)


class ExpOp(operator.Operator,
            implements=_registry.FunctionNames.EXP,
            symbol="exp"):
    def forward(self, x: variable.Variable):
        out = variable.Variable(np.exp(get_data(x)))
        self.save_for_backward("out", out if self.requires_grad else None)
        return out

    def backward(self, grad_output):
        out = self.saved_value("out")
        return MulOp.apply(out, grad_output) if out is not None else None


class LogOp(operator.Operator,
            implements=_registry.FunctionNames.LOG,
            symbol="ln"):
    def forward(self, x: variable.Variable):
        out = variable.Variable(np.log(get_data(x)))
        self.save_for_backward("x", x if x.requires_grad else None)
        return out

    def backward(self, grad_output: variable.Variable):
        x = self.saved_value("x")
        return MulOp.apply(PowOp.apply(x, -1),
                           grad_output) if x is not None else None


class PowOp(operator.Operator,
            implements=_registry.FunctionNames.POW,
            symbol="**"):
    def forward(self, x1: variable.Variable, x2: variable.Variable):
        out = variable.Variable(np.power(get_data(x1), get_data(x2)))
        self.save_for_backward(
            "x1", x1 if operator.prop_grad(
                x1) or operator.prop_grad(x2) else None)
        self.save_for_backward(
            "x2", x2 if operator.prop_grad(x1) else None)
        self.save_for_backward(
            "out", out if operator.prop_grad(x2) else None)
        return out

    def backward(self, grad_output: variable.Variable):
        x2 = self.saved_value("x2")
        x1 = self.saved_value("x1")
        out = self.saved_value("out")
        inp_grad = x2_grad = None
        if x1 is not None:
            inp_grad = (
                MulOp.apply(
                    MulOp.apply(x2, PowOp.apply(
                        x1, MinusOp.apply(x2, 1))), grad_output))
        if out is not None:
            x2_grad = MulOp.apply(out, LogOp.apply(x1))
        return inp_grad, x2_grad


class SigmoidOp(operator.Operator,
                implements=_registry.FunctionNames.SIGMOID,
                symbol="<&sigma;>"):
    def forward(self, x: variable.Variable):
        PowOp.apply(AddOp.apply(1, ExpOp.apply(NegOp.apply(x))), -1)
        out = (1 + (-x).exp())**-1
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output: variable.Variable):
        out = self.saved_value("out")
        return MulOp.apply(
            MulOp.apply(out, MinusOp.apply(1, out)), grad_output)


class TanhOp(operator.Operator,
             implements=_registry.FunctionNames.TANH,
             symbol="tanh"):
    def forward(self, x: variable.Variable):
        out = variable.Variable(np.tanh(get_data(x)))
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output: variable.Variable):
        out = self.saved_value("out")
        return MulOp.apply(MinusOp(1, PowOp.apply(out, 2), grad_output))
