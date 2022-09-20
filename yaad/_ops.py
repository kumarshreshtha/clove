from __future__ import annotations

import math
import numbers
from typing import Optional, Union

from yaad import operator
from yaad import variable
from yaad import _registry


def is_number(input):
    return isinstance(input, numbers.Number)


class CloneOp(operator.Operator,
              implements=_registry.FunctionNames.CLONE,
              symbol="clone"):
    def forward(self, input: variable.Variable):
        return input.data

    def backward(self, grad_output: variable.Variable):
        return grad_output


class AddOp(operator.Operator,
            implements=_registry.FunctionNames.ADD,
            symbol="+"):
    def forward(self,
                input: variable.Variable,
                other: Union[variable.Variable, numbers.Number]):
        """Adds two nodes or a node and a number."""
        input_data = input if is_number(input) else input.data
        other_data = other if is_number(other) else other.data
        return variable.Variable(input_data + other_data)

    def backward(self, grad_output: Optional[variable.Variable]):
        if grad_output is None:
            return None, None
        return grad_output, grad_output


class MulOp(operator.Operator,
            implements=_registry.FunctionNames.MULTIPLY,
            symbol="<&times;>"):
    def forward(self,
                input: variable.Variable,
                other: variable.Variable):
        input_data = input if is_number(input) else input.data
        other_data = other if is_number(other) else other.data
        self.save_for_backward("other", other)
        self.save_for_backward("input", input)
        return variable.Variable(input_data * other_data)

    def backward(self, grad_output: variable.Variable):
        other = self.saved_value("other")
        input = self.saved_value("input")
        grad_input = other * grad_output
        grad_other = input * grad_output
        return grad_input, grad_other


class NegOp(operator.Operator,
            implements=_registry.FunctionNames.NEGATE,
            symbol="-1*"):
    def forward(self, input):
        return input * -1

    def backward(self, grad_output):
        return -grad_output


class MinusOp(operator.Operator,
              implements=_registry.FunctionNames.SUBTRACT,
              symbol="-"):
    def forward(self, input, other):
        return input + (-other)

    def backward(self, grad_output: variable.Variable):
        return grad_output, -grad_output


class ExpOp(operator.Operator,
            implements=_registry.FunctionNames.EXP,
            symbol="exp"):
    def forward(self, input: variable.Variable):
        out = variable.Variable(math.exp(input.data))
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output):
        return self.saved_value("out") * grad_output


class PowOp(operator.Operator,
            implements=_registry.FunctionNames.POW,
            symbol="**"):
    def forward(self, input: variable.Variable, other: numbers.Number):
        out = input.data**other
        self.save_for_backward("input", input)
        self.save_for_backward("other", other)
        return out

    def backward(self, grad_output: variable.Variable):
        other = self.saved_value("other")
        input = self.saved_value("input")
        return other * input**(other - 1) * grad_output


class SigmoidOp(operator.Operator,
                implements=_registry.FunctionNames.SIGMOID,
                symbol="<&sigma;>"):
    def forward(self, input: variable.Variable):
        out = (1 + -input.exp())**-1
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output: variable.Variable):
        out = self.saved_value("out")
        return grad_output * out * (1 - out)
