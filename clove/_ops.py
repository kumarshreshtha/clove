from __future__ import annotations

import numbers
import numpy as np
from typing import Optional, Union

from clove import operator
from clove import variable
from clove import _registry


def is_number(input):
    return isinstance(input, numbers.Number)


def get_data(array: Union[variable.Variable, np.ndarray, numbers.Number]):
    if isinstance(array, variable.Variable):
        return array.data
    return array

# TODO: update function signatures automatically from backend docs.


class CloneOp(operator.Operator,
              implements=_registry.FunctionNames.CLONE,
              symbol="clone"):
    def forward(self, input: variable.Variable):
        return variable.Variable(input.data.copy())

    def backward(self, grad_output: variable.Variable):
        return grad_output


class TransposeOp(operator.Operator,
                  implements=_registry.FunctionNames.TRANSPOSE,
                  symbol="T"):

    def forward(self, input: variable.Variable):
        return variable.Variable(get_data(input).T)

    def backward(self, grad_output: variable.Variable):
        return TransposeOp.apply(grad_output)


class AddOp(operator.Operator,
            implements=_registry.FunctionNames.ADD,
            symbol="+"):
    def forward(self,
                input: variable.Variable,
                other: Union[variable.Variable, numbers.Number]):
        """Adds two nodes or a node and a number."""
        return variable.Variable(np.add(get_data(input), get_data(other)))

    def backward(self, grad_output: Optional[variable.Variable]):
        return grad_output, grad_output


class MulOp(operator.Operator,
            implements=_registry.FunctionNames.MULTIPLY,
            symbol="<&times;>"):
    def forward(self,
                input: variable.Variable,
                other: variable.Variable):
        self.save_for_backward(
            "other", other if operator.prop_grad(input) else None)
        self.save_for_backward(
            "input", input if operator.prop_grad(other) else None)
        return variable.Variable(np.multiply(get_data(input), get_data(other)))

    def backward(self, grad_output: variable.Variable):
        other = self.saved_value("other")
        input = self.saved_value("input")
        grad_input = MulOp.apply(
            other, grad_output) if other is not None else None
        grad_other = MulOp.apply(
            input, grad_output) if input is not None else None
        return grad_input, grad_other


class MatmulOp(operator.Operator,
               implements=_registry.FunctionNames.MATMUL,
               symbol="@"):
    def forward(self,
                input: variable.Variable,
                other: variable.Variable):
        self.save_for_backward(
            "other", other if operator.prop_grad(input) else None)
        self.save_for_backward(
            "input", input if operator.prop_grad(other) else None)
        return variable.Variable(np.matmul(get_data(input), get_data(other)))

    def backward(self, grad_output: variable.Variable):
        other = self.saved_value("other")
        input = self.saved_value("input")
        grad_input = (MatmulOp.apply(grad_output, TransposeOp.apply(other))
                      if other is not None else None)
        grad_other = (MatmulOp.apply(TransposeOp.apply(input), grad_output)
                      if input is not None else None)
        return grad_input, grad_other


class NegOp(operator.Operator,
            implements=_registry.FunctionNames.NEGATE,
            symbol="-1*"):
    def forward(self, input):
        return variable.Variable(np.negative(get_data(input)))

    def backward(self, grad_output):
        return NegOp.apply(grad_output)


class MinusOp(operator.Operator,
              implements=_registry.FunctionNames.SUBTRACT,
              symbol="-"):
    def forward(self, input, other):
        return variable.Variable(np.subtract(get_data(input), get_data(other)))

    def backward(self, grad_output: variable.Variable):
        return grad_output, NegOp.apply(grad_output)


class ExpOp(operator.Operator,
            implements=_registry.FunctionNames.EXP,
            symbol="exp"):
    def forward(self, input: variable.Variable):
        out = variable.Variable(np.exp(get_data(input)))
        self.save_for_backward("out", out if self.requires_grad else None)
        return out

    def backward(self, grad_output):
        out = self.saved_value("out")
        return MulOp.apply(out, grad_output) if out is not None else None


class LogOp(operator.Operator,
            implements=_registry.FunctionNames.LOG,
            symbol="ln"):
    def forward(self, input: variable.Variable):
        out = variable.Variable(np.log(get_data(input)))
        self.save_for_backward("input", input if input.requires_grad else None)
        return out

    def backward(self, grad_output: variable.Variable):
        input = self.saved_value("input")
        return MulOp.apply(PowOp.apply(input, -1),
                           grad_output) if input is not None else None


class PowOp(operator.Operator,
            implements=_registry.FunctionNames.POW,
            symbol="**"):
    def forward(self, input: variable.Variable, other: variable.Variable):
        out = variable.Variable(np.power(get_data(input), get_data(other)))
        self.save_for_backward(
            "input", input if operator.prop_grad(
                input) or operator.prop_grad(other) else None)
        self.save_for_backward(
            "other", other if operator.prop_grad(input) else None)
        self.save_for_backward(
            "out", out if operator.prop_grad(other) else None)
        return out

    def backward(self, grad_output: variable.Variable):
        other = self.saved_value("other")
        input = self.saved_value("input")
        out = self.saved_value("out")
        inp_grad = other_grad = None
        if input is not None:
            inp_grad = (
                MulOp.apply(
                    MulOp.apply(other, PowOp.apply(
                        input, MinusOp.apply(other, 1))), grad_output))
        if out is not None:
            other_grad = MulOp.apply(out, LogOp.apply(input))
        return inp_grad, other_grad


class SigmoidOp(operator.Operator,
                implements=_registry.FunctionNames.SIGMOID,
                symbol="<&sigma;>"):
    def forward(self, input: variable.Variable):
        PowOp.apply(AddOp.apply(1, ExpOp.apply(NegOp.apply(input))), -1)
        out = (1 + (-input).exp())**-1
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output: variable.Variable):
        out = self.saved_value("out")
        return MulOp.apply(
            MulOp.apply(out, MinusOp.apply(1, out)), grad_output)


class TanhOp(operator.Operator,
             implements=_registry.FunctionNames.TANH,
             symbol="tanh"):
    def forward(self, input: variable.Variable):
        out = variable.Variable(np.tanh(get_data(input)))
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output: variable.Variable):
        out = self.saved_value("out")
        return MulOp.apply(MinusOp(1, PowOp.apply(out, 2), grad_output))
