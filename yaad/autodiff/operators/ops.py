from __future__ import annotations

import math
import numbers
from typing import Optional, Union

from yaad import grad_mode
from yaad.autodiff import engine
from yaad.autodiff import node
from yaad.autodiff.operators import base


def is_number(inp):
    return isinstance(inp, numbers.Number)


class CloneOp(base.Operator, fn_name="clone", symbol="clone"):
    def forward(self, inp: node.Node):
        return node.Node(inp.data)

    def backward(self, grad_output: node.Node):
        return grad_output


class AddOp(base.Operator, fn_name="add", symbol="+"):
    def forward(self,
                inp: node.Node,
                other: Union[node.Node, numbers.Number]):
        """Adds two nodes or a node and a number."""
        inp_data = inp if is_number(inp) else inp.data
        other_data = other if is_number(other) else other.data
        return node.Node(inp_data + other_data)

    def backward(self, grad_output: Optional[node.Node]):
        if grad_output is None:
            return None, None
        return grad_output, grad_output


class MulOp(base.Operator, fn_name="multiply", symbol="<&times;>"):
    def forward(self,
                inp: node.Node,
                other: node.Node):
        inp_data = inp if is_number(inp) else inp.data
        other_data = other if is_number(other) else other.data
        self.save_for_backward("other", other)
        self.save_for_backward("inp", inp)
        return node.Node(inp_data * other_data)

    def backward(self, grad_output: node.Node):
        other = self.saved_value("other")
        inp = self.saved_value("inp")
        grad_inp = other * grad_output
        grad_other = inp * grad_output
        return grad_inp, grad_other


class MinusOp(base.Operator, fn_name="subtract", symbol="-"):
    def forward(self, inp, other):
        return inp + (-other)

    def backward(self, grad_output: node.Node):
        return grad_output, -grad_output


class ExpOp(base.Operator, fn_name="exp", symbol="e"):
    def forward(self, inp: node.Node):
        out = node.Node(math.exp(inp.data))
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output):
        return self.saved_value("out") * grad_output


class PowOp(base.Operator, fn_name="pow", symbol="**"):
    def forward(self, inp: node.Node, other: numbers.Number):
        out = node.Node(inp.data**other)
        self.save_for_backward("inp", inp)
        self.save_for_backward("other", other)
        return out

    def backward(self, grad_output: node.Node):
        other = self.saved_value("other")
        inp = self.saved_value("inp")
        return other * inp**(other - 1) * grad_output


class SigmoidOp(base.Operator, fn_name="sigmoid", symbol="<&sigma;>"):
    def forward(self, inp: node.Node):
        out = (1 + ExpOp.apply(-inp))**-1
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output: node.Node):
        out = self.saved_value("out")
        return grad_output * out * (1 - out)
