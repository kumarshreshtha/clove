import numbers
from typing import TYPE_CHECKING, Optional, Union
import warnings

from clove import autodiff
from clove import ops
from clove import binding_utils
from clove import backend as backend_lib

if TYPE_CHECKING:
    from clove import operator


class Variable:
    """Container class for a variable and it's gradient."""

    def __init__(self,
                 data,
                 backend: backend_lib.Backend,
                 requires_grad=False,
                 name: str = None):
        if isinstance(data, numbers.Number):
            data = backend.make_from_number(data)
        self._data = data
        self._grad = None
        self.requires_grad = requires_grad
        self.name = name
        self.op: Optional[operator.Operator] = None
        self._backend = backend
        self._retains_grad = False

    @property
    def data(self):
        return self._data

    @property
    def backend(self):
        return self._backend

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

    def backward(self, grad_output=None, retain_graph=False):
        autodiff.backward(self, grad_output, retain_graph)

    def zero_grad(self, set_to_none: bool = False):
        self._grad = None if set_to_none else 0.

    def __setitem__(self, key, value):
        if not self.is_leaf() or self.requires_grad:
            raise RuntimeError("cannot mutate variable attached to a graph"
                               " in place.")
        self.data[key] = value

    # TODO: format this better.
    def __repr__(self):
        grad_repr = f", requires_grad=True" if self.requires_grad else ""
        return f"{self.__class__.__name__}({self.data}{grad_repr})"

    # TODO: add resolve device.
    @property
    def shape(self):
        return self.backend.resolve_shape(self.data)

    @property
    def dtype(self):
        return self.backend.resolve_dtype(self.data)

    @property
    def numel(self):
        return self.backend.resolve_numel(self.data)

    def is_scalar(self):
        return self.shape == tuple() or self.shape == (1,)

    __getitem__ = binding_utils.make_method("__getitem", ops.IndexOp)
    __add__ = binding_utils.make_method("__add__", ops.AddOp)
    __radd__ = binding_utils.make_method("__radd__", ops.AddOp)
    __mul__ = binding_utils.make_method("__mul__", ops.MulOp)
    __rmul__ = binding_utils.make_method("__rmul__", ops.MulOp)
    __truediv__ = binding_utils.make_method("__div__", ops.DivOp)
    __sub__ = binding_utils.make_method("__sub__", ops.MinusOp)
    __neg__ = binding_utils.make_method("__neg__", ops.NegOp)
    __pow__ = binding_utils.make_method("__pow__", ops.PowOp)
    __matmul__ = binding_utils.make_method("__matmul__", ops.MatmulOp)
    reciprocal = binding_utils.make_method("__matmul__", ops.ReciprocalOp)
    sigmoid = binding_utils.make_method("sigmoid", ops.SigmoidOp)
    exp = binding_utils.make_method("exp", ops.ExpOp)
    log = binding_utils.make_method("log", ops.LogOp)
    tanh = binding_utils.make_method("tanh", ops.TanhOp)
    transpose = binding_utils.make_method("transpose", ops.TransposeOp)
    permute = binding_utils.make_method("permute", ops.PermuteOp)
    sum = binding_utils.make_method("sum", ops.SumOp)
    prod = binding_utils.make_method("prod", ops.ProdOp)
    mean = binding_utils.make_method("mean", ops.MeanOp)
    reshape = binding_utils.make_method("reshape", ops.ReshapeOp)
    expand = binding_utils.make_method("expand", ops.ExpandOp)

    def __rsub__(self, other):
        return other + (-self)

    def __rtruediv__(self, other):
        return ops.DivOp.apply(other, self)

    @property
    def T(self):
        return self.transpose()


ArrayOrScalar = Union[
    numbers.Number,
    Variable,
]
