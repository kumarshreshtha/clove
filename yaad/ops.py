from __future__ import annotations

import abc
import dataclasses
import math
from typing import Optional, Sequence, Union
import weakref

from yaad import node
from yaad import grad_mode

Number = Union[int, float]


def prop_grad(inp):
    return isinstance(inp, node.Node) and inp.requires_grad


@dataclasses.dataclass
class GradStore:
    value: node.Node = None

    def reset(self):
        self.value = None

    def update(self, grad: node.Node):
        # Note: if incoming grad has requires_grad=True then this value becomes
        # part of the graph thus making higher order derivatives possible.
        self.value = self.value + grad if self.value is not None else grad


class Operator(abc.ABC):

    def __init__(self,
                 children: Sequence[Operator] = (),
                 requires_grad: bool = False,
                 variable: node.Node = None):
        self._cache = {}
        self._children = tuple(children)
        self.requires_grad = requires_grad
        self._var_ref = weakref.ref(variable) if variable is not None else None
        self.grad_store = GradStore()

    def __init_subclass__(cls, symbol: Optional[str] = None) -> None:
        cls.symbol = symbol if symbol is not None else cls.__name__

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


class CloneOp(Operator):
    def forward(self, inp: node.Node):
        return node.Node(inp.data)

    def backward(self, grad_output: node.Node):
        return grad_output

# TODO: think how to handle mixed ops between Node and int/float.


class AddOp(Operator, symbol="+"):
    def forward(self,
                inp: node.Node,
                other: node.Node):
        return node.Node(inp.data + other.data)

    def backward(self, grad_output: Optional[node.Node]):
        if grad_output is None:
            return None, None
        return grad_output, grad_output


class MulOp(Operator, symbol="<&times;>"):
    def forward(self,
                inp: node.Node,
                other: node.Node):
        self.save_for_backward("other", other)
        self.save_for_backward("inp", inp)
        return node.Node(inp.data * other.data)

    def backward(self, grad_output: node.Node):
        other = self.saved_value("other")
        inp = self.saved_value("inp")
        grad_inp = other * grad_output
        grad_other = inp * grad_output
        return grad_inp, grad_other


class MinusOp(Operator, symbol="-"):
    def forward(self, inp, other):
        return node.Node(inp.data - other.data)

    def backward(self, grad_output: node.Node):
        return grad_output, -grad_output


class ExpOp(Operator, symbol="e"):
    def forward(self, inp: node.Node):
        out = node.Node(math.exp(inp.data))
        self.save_for_backward("out", out)

    def backward(self, grad_output):
        return self.saved_value("out") * grad_output


class PowOp(Operator, symbol=""):
    def forward(self, inp: node.Node, other: Number):
        out = node.Node(inp.data**other)
        self.save_for_backward("inp", inp)
        self.save_for_backward("other", other)
        return out

    def backward(self, grad_output: node.Node):
        other = self.saved_value("other")
        inp = self.saved_value("inp")
        return other * inp**(other - 1) * grad_output


class SigmoidOp(Operator, symbol="<&sigma;>"):
    def forward(self, inp: node.Node):
        out = (1 + exp(-inp))**-1
        self.save_for_backward("out", out)
        return out

    def backward(self, grad_output: node.Node):
        out = self.saved_value(out)
        return grad_output * out * (1 - out)


def add(input: node.Node, other: Union[node.Node, Number]):
    return AddOp.apply(input, other)


def multiply(input: node.Node, other: Union[node.Node, Number]):
    return MulOp.apply(input, other)


def clone(input: node.Node):
    return CloneOp.apply(input)


def exp(input: node.Node):
    return ExpOp.apply(input)
