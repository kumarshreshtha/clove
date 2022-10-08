import numbers
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union
import warnings

from clove import autodiff
from clove import ops
from clove import bindings_utils
from clove import _backend

if TYPE_CHECKING:
    from clove import operator

# TODO: variable should have properties dtype, shape, size, numel etc
# must form associations with the backend and not access them directly.
# as that might break the compatibility with backend.


class Variable:
    """Container class for a variable and it's gradient."""

    METHODS_FROM_OPS = dict(
        __add__=ops.AddOp,
        __radd__=ops.AddOp,
        __mul__=ops.MulOp,
        __rmul__=ops.MulOp,
        __sub__=ops.MinusOp,
        __neg__=ops.NegOp,
        __pow__=ops.PowOp,
        __matmul__=ops.MatmulOp,
        sigmoid=ops.SigmoidOp,
        exp=ops.ExpOp,
        tanh=ops.TanhOp,
        transpose=ops.TransposeOp
    )

    def __init__(self,
                 data,
                 requires_grad=False,
                 name: str = None):

        self._data = data
        self._grad = None
        self.requires_grad = requires_grad
        self.name = name
        self.op: Optional[operator.Operator] = None
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

    def __rsub__(self, other):
        return other + (-self)

    @property
    def T(self):
        return self.transpose()

    def is_leaf(self):
        return self.op is None

    def backward(self, grad_output=None, retain_graph=False):
        autodiff.backward(self, grad_output, retain_graph)

    def zero_grad(self, set_to_none: bool = False):
        self._grad = None if set_to_none else 0.

    def __repr__(self):
        grad_repr = f", requires_grad=True" if self.requires_grad else ""
        return f"{self.__class__.__name__}({self.data}{grad_repr})"


for __method_name, __op in Variable.METHODS_FROM_OPS.items():
    setattr(Variable,
            __method_name,
            bindings_utils.make_method(__method_name, __op))

ArrayLike = Union[
    numbers.Number,
    Sequence[numbers.Number],
    Sequence[Sequence[Any]],
    Variable,
]
