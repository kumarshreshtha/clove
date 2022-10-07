from typing import TYPE_CHECKING, Optional
import warnings

from clove import autodiff
from clove import definitions

if TYPE_CHECKING:
    from clove import operator

# TODO: variable should have properties dtype, shape, size, numel etc
# must form associations with the backend and not access them directly.
# as that might break the compatibility with backend.


class Variable:
    """Container class for a variable and it's gradient."""

    METHODS_FROM_DEFN = dict(
        __add__=definitions.Function.ADD,
        __radd__=definitions.Function.ADD,
        __mul__=definitions.Function.MULTIPLY,
        __rmul__=definitions.Function.MULTIPLY,
        __sub__=definitions.Function.SUBTRACT,
        __neg__=definitions.Function.NEGATIVE,
        __pow__=definitions.Function.POW,
        __matmul__=definitions.Function.MATMUL,
        sigmoid=definitions.Function.SIGMOID,
        exp=definitions.Function.EXP,
        tanh=definitions.Function.TANH,
        transpose=definitions.Function.TRANSPOSE
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
