from __future__ import annotations

import abc
import contextlib
import dataclasses
from typing import Optional, Sequence, Union
import weakref

from yaad import node

Number = Union[int, float]


@contextlib.contextmanager
def disable_grad():
    try:
        Operator.compute_grad = False
        yield
    finally:
        Operator.compute_grad = True


def prop_grad(inp):
    return isinstance(inp, node.Node) and inp.requires_grad


@dataclasses.dataclass
class GradStore:
    value: float = 0.

    def reset(self):
        self.value = 0.

    def update(self, grad):
        self.value = self.value + grad


class Operator(abc.ABC):
    compute_grad = False

    def __init__(self,
                 variable: node.Node,
                 _children: Sequence[Operator] = (),
                 requires_grad: bool = False):
        self._cache = {}
        self._children = tuple(_children)
        self.requires_grad = requires_grad
        self._var_ref = weakref.ref(variable)
        self.grad_store = GradStore()

    def __init_subclass__(cls, symbol: Optional[str] = None) -> None:
        cls.symbol = symbol if symbol is not None else cls.__name__

    @property
    def next_ops(self):
        return self._children

    def delete_edges(self):
        self._children = tuple()

    @property
    def variable(self):
        return self._var_ref()

    @abc.abstractmethod
    def forward(self, *args, out: node.Node):
        raise NotImplementedError

    @abc.abstractmethod
    def backward(self, grad_output: node.Node):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args):
        children = []
        requires_grad = cls.compute_grad
        if requires_grad:
            for arg in args:
                if prop_grad(arg):
                    op = LeafOp(arg) if arg._op is None else arg._op
                    children.append(op)
                else:
                    children.append(None)
            requires_grad = any([child is not None for child in children])
        op = cls(children=children if requires_grad else [],
                 requires_grad=requires_grad)
        out = node.Node(data=None,
                        requires_grad=requires_grad,
                        _op=op if requires_grad else None)
        # ops have their own backward. therefore the operations they do within
        #  their forward should be detached from the computation graph.
        with disable_grad():
            op.forward(*args, out=out)
        return out

    def save_for_backward(self, name, value):
        # TODO: maybe should save a copy of the Node?
        # if it is mutated inplace after that could lead to problems.
        if self.requires_grad:
            self._cache[name] = value

    def saved_value(self, name):
        return self._cache.get(name, None)


class LeafOp(Operator, symbol="leaf"):

    def __init__(self, variable):
        super().__init__(variable, requires_grad=variable.requires_grad)
        self.symbol = (
            variable.name if variable.name is not None else self.symbol)

    def forward(self, inp: node.Node):
        return inp

    def backward(self, grad_output: node.Node):
        return grad_output


class CloneOp(Operator):
    def forward(self, inp: node.Node, out: node.Node):
        out.data = inp.data

    def backward(self, grad_output: node.Node):
        return grad_output

# TODO: think how to handle mixed ops between Node and int/float.


class AddOp(Operator, symbol="+"):
    def forward(self,
                inp: node.Node,
                other: node.Node,
                out: node.Node):
        out.data = inp.data + other.data

    def backward(self, grad_output: Optional[node.Node]):
        if grad_output is None:
            return None, None
        return grad_output, grad_output


class MulOp(Operator, symbol="*"):
    def forward(self,
                inp: node.Node,
                other: node.Node,
                out: node.Node):
        if not out.requires_grad:
            out.data = inp.data * other.data
            return
        if inp.requires_grad:
            self.save_for_backward("other", inp)
        if other.requires_grad:
            self.save_for_backward("inp", other)
        out.data = inp.data * other.data

    def backward(self, grad_output: node.Node):
        # reusing the forward definition allows for higher order derivatives.
        other = self.saved_value("other")
        inp = self.saved_value("inp")
        grad_inp = other * grad_output if other is not None else None
        grad_other = inp * grad_output if inp is not None else None
        return grad_inp, grad_other


def add(input: node.Node, other: Union[node.Node, Number]):
    return AddOp.apply(input, other)


def multiply(input: node.Node, other: Union[node.Node, Number]):
    return MulOp.apply(input, other)


def clone(input: node.Node):
    return CloneOp.apply(input)
