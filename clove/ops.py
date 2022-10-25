from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

from clove import operator

if TYPE_CHECKING:
    from clove import variable


def resolve_dims_for_reduction(dims, total_dims):
    if isinstance(dims, int):
        return resolve_dims_for_reduction((dims, ), total_dims)
    if dims is None:
        return tuple(range(total_dims))
    return tuple(total_dims + d if d < 0 else d for d in dims)

# TODO: throw an error when shapes don't match for the trailing dims.
# TODO: how to handle non arrays?
# should we convert to arrays implicitely?


def broadcast_shapes(x1, x2):
    x1_shape = [1] * max(len(x2.shape) - len(x1.shape), 0) + list(x1.shape)
    x2_shape = [1] * max(len(x1.shape) - len(x2.shape), 0) + list(x2.shape)
    ndims = len(x1_shape)
    common_shape = []
    x1_reduction_dims = []
    x2_reduction_dims = []
    for i, (d1, d2) in enumerate(reversed(list(zip(x1_shape, x2_shape)))):
        if d1 == d2:
            common_shape.append(d1)
        elif d1 == 1 and d2 != 1:
            common_shape.append(d2)
            x1_reduction_dims.append(ndims - i - 1)
        elif d2 == 1 and d1 != 1:
            common_shape.append(d1)
            x2_reduction_dims.append(ndims - i - 1)
        else:
            raise ValueError(
                f"Cannot broadcast operands with shape {x1.shape} {x2.shape}")
    common_shape.reverse()
    return tuple(common_shape), x1_reduction_dims, x2_reduction_dims


def resolve_shape_for_expansion(new_shape, old_shape):
    leading_dims = len(new_shape) - len(old_shape)
    if any([d == -1 for d in new_shape[:leading_dims]]):
        raise RuntimeError("expanded size of the tensor (-1) isn't allowed in"
                           " a leading, non-existing dimension")
    return (new_shape[:leading_dims] +
            tuple([old_shape[i] if d == -1 else d
                   for i, d in enumerate(new_shape[leading_dims:])]))

# TODO: account for 1 in shape dims.


class ExpandOp(operator.Operator, fn_name="expand"):
    def forward(self,
                x: variable.Variable,
                shape: Union[int, Tuple[int, ...]]):
        shape = resolve_shape_for_expansion(shape, x.shape)
        self._cache.reduction_dim = tuple(range(0, len(shape) - len(x.shape)))
        return self.evaluate(x, shape)

    def backward(self, grad_out):
        if not self._cache.reduction_dim:
            return grad_out
        return SumOp.apply(grad_out, dim=self._cache.reduction_dim)

# TODO: add support for keep_dims.
# TODO: keep track of dim to know what dimension to expand in.


def resolve_shape(x, shape: Sequence[int]):
    shape = list(shape)
    idx_minus_one = None
    numel = 1
    for i, s in enumerate(shape):
        if s == -1:
            if idx_minus_one is not None:
                raise ValueError()
            idx_minus_one = i
        else:
            numel *= s
    if idx_minus_one is not None:
        if x.numel % numel:
            raise ValueError(
                f"cannot reshape array of size {x.numel} into shape {shape}")
        shape[shape.index(-1)] = x.numel // numel
    return shape


class ReshapeOp(operator.Operator, fn_name="squeeze"):
    def forward(self,
                x: variable.Variable,
                shape: Tuple[int, ...]):
        shape = resolve_shape(x, shape)
        self._cache.shape = x.shape
        return self.evaluate(x, shape)

    def backward(self, grad_output: variable.Variable):
        return ReshapeOp.apply(grad_output, self._cache.shape)


def expand_dims(shape, dims):
    return [s if i not in dims else 1 for i, s in shape]


class SumOp(operator.Operator, fn_name="sum"):

    def forward(self,
                x: variable.Variable,
                dim: Union[int, Tuple[int, ...], None] = None
                ) -> variable.Variable:
        dim = resolve_dims_for_reduction(dim, len(x.shape))
        self._cache.dim = dim
        self._cache.shape = x.shape
        return self.evaluate(x, dim)

    def backward(self, grad_out):
        expanded_shape = expand_dims(self._cache.shape, self._cache.dim)
        return ExpandOp.apply(
            ReshapeOp.apply(grad_out, expanded_shape), self._cache.shape)


class MeanOp(operator.Operator, fn_name="mean"):
    def forward(self,
                x: variable.Variable,
                dim: Union[int, Tuple[int, ...], None] = None
                ) -> variable.Variable:
        dim = resolve_dims_for_reduction(dim, len(x.shape))
        self._cache.shape = x.shape
        self._cache.dim = dim
        self._cache.div = 1 / math.prod([x.shape[d] for d in dim])
        return self.evaluate(x, dim)

    def backward(self, grad_out):
        expanded_shape = expand_dims(self._cache.shape, self._cache.dim)
        grad_out = ExpandOp.apply(ReshapeOp.apply(grad_out, expanded_shape),
                                  self._cache.shape)
        return MulOp.apply(grad_out, self._cache.div)


class ProdOp(operator.Operator, fn_name="prod"):
    def forward(self,
                x: variable.Variable,
                dim: Union[int, Tuple[int, ...], None] = None
                ) -> variable.Variable:
        dim = resolve_dims_for_reduction(dim, len(x.shape))
        out = self.evaluate(x, dim)
        self._cache.x = x
        self._cache.out = out
        self._cache.dim = dim
        return out

    def backward(self, grad_output):
        out = self._cache.out
        x = self._cache.x
        expanded_shape = expand_dims(x.shape, self._cache.dim)
        grad = DivOp.apply(ReshapeOp.apply(out, expanded_shape), x)
        return MulOp.apply(grad, grad_output)


class CloneOp(operator.Operator, fn_name="clone"):
    def forward(self, x: variable.ArrayOrScalar) -> variable.Variable:
        return self.evaluate(x)

    def backward(self, grad_out: variable.Variable):
        return grad_out


class TransposeOp(operator.Operator, fn_name="transpose", symbol="T"):

    def forward(self, x: variable.ArrayOrScalar, dim_0: int, dim_1: int):
        self._cache.d0 = dim_0
        self._cache.d1 = dim_1
        return self.evaluate(x, dim_0, dim_1)

    def backward(self, grad_out: variable.Variable):
        return TransposeOp.apply(grad_out, self._cache.d1, self._cache.d0)


class PermuteOp(operator.Operator, fn_name="permute"):

    def forward(self, x: variable.ArrayOrScalar, dim: Sequence[int]):
        if self.requires_grad:
            self._cache.rev_dim = sorted(range(len(dim)), key=dim.__getitem__)
        return self.evaluate(x, dim)

    def backward(self, grad_out: variable.Variable):
        return PermuteOp.apply(grad_out, self._cache.rev_dim)

# TODO: account for broadcasting of arrays

# TODO: check ArrayOrScalar to tensor or scalar, nothing else.


class AddOp(operator.Operator, fn_name="add", symbol="+"):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        if isinstance(x1, variable.Variable) and isinstance(
                x2, variable.Variable) and x1.shape != x2.shape:
            common_shape, x1_reduction_dim, x2_reduction_dim = (
                broadcast_shapes(x1, x2))
            x1 = ExpandOp.apply(
                x1, common_shape) if x1.shape != common_shape else x1
            x2 = ExpandOp.apply(
                x2, common_shape) if x2.shape != common_shape else x2
            if x1_reduction_dim:
                self._cache.x1_reduction_dim = x1_reduction_dim
            if x2_reduction_dim:
                self._cache.x2_reduction_dim = x2_reduction_dim
        return self.evaluate(x1, x2)

    def backward(self, grad_out: Optional[variable.ArrayOrScalar]):
        x1_grad = x2_grad = grad_out
        if 'x1_reduction_dim' in self._cache:
            x1_grad = SumOp.apply(grad_out, self._cache.x1_reduction_dim)
        if 'x2_reduction_dim' in self._cache:
            x2_grad = SumOp.apply(grad_out, self._cache.x2_reduction_dim)
        return x1_grad, x2_grad


class MulOp(operator.Operator, fn_name="multiply", symbol="<&times;>"):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        common_shape, x1_reduction_dim, x2_reduction_dim = (
            broadcast_shapes(x1, x2))
        x1 = ExpandOp.apply(
            x1, common_shape) if x1.shape != common_shape else x1
        x2 = ExpandOp.apply(
            x2, common_shape) if x2.shape != common_shape else x2
        if x1_reduction_dim:
            self._cache.x1_reduction_dim = x1_reduction_dim
        if x2_reduction_dim:
            self._cache.x2_reduction_dim = x2_reduction_dim
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = MulOp.apply(x2, grad_out) if x2 is not None else None
        grad_x2 = MulOp.apply(x1, grad_out) if x1 is not None else None
        if 'x1_reduction_dim' in self._cache:
            grad_x1 = SumOp.apply(grad_x1, self._cache.x1_reduction_dim)
        if 'x2_reduction_dim' in self._cache:
            grad_x2 = SumOp.apply(grad_x2, self._cache.x2_reduction_dim)
        return grad_x1, grad_x2


class MinusOp(operator.Operator, fn_name="subtract", symbol="-"):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.Variable):
        return grad_out, NegOp.apply(grad_out)


class DivOp(operator.Operator, fn_name="divide"):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = grad_x2 = None
        if x2 is not None:
            grad_x1 = MulOp.apply(ReciprocalOp.apply(x2), grad_out)
        if x1 is not None:
            grad_x2 = (
                MulOp.apply(
                    MulOp.apply(
                        NegOp.apply(PowOp.apply(ReciprocalOp.apply(x2), 2)),
                        x1
                    ),
                    grad_out
                )
            )
        return grad_x1, grad_x2


class MatmulOp(operator.Operator, fn_name="matmul", symbol="@"):
    def forward(self, x1: variable.Variable, x2: variable.Variable):
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = (MatmulOp.apply(grad_out, TransposeOp.apply(x2))
                   if x2 is not None else None)
        grad_x2 = (MatmulOp.apply(TransposeOp.apply(x1), grad_out)
                   if x1 is not None else None)
        return grad_x1, grad_x2


class PowOp(operator.Operator, fn_name="power", symbol="**"):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        out = self.evaluate(x1, x2)
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


class ReciprocalOp(operator.Operator, fn_name="reciprocal"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.Variable):
        out_sq = NegOp.apply(PowOp.apply(self._cache.out, 2))
        return MulOp.apply(grad_out, out_sq)

# TODO: maybe just use the variable ops in backward. It's cleaner.


class NegOp(operator.Operator, fn_name="negative", symbol="-1*"):
    def forward(self, x: variable.ArrayOrScalar):
        return self.evaluate(x)

    def backward(self, grad_out):
        return NegOp.apply(grad_out)


class ExpOp(operator.Operator, fn_name="exp", symbol="exp"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out):
        out = self._cache.out
        return MulOp.apply(out, grad_out) if out is not None else None


class LogOp(operator.Operator, fn_name="log", symbol="ln"):
    def forward(self, x: variable.ArrayOrScalar):
        self._cache.x = x
        return self.evaluate(x)

    def backward(self, grad_out: variable.Variable):
        x = self._cache.x
        return (MulOp.apply(PowOp.apply(x, -1), grad_out)
                if x is not None else None)


class SigmoidOp(operator.Operator, fn_name="sigmoid", symbol="<&sigma;>"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.Variable):
        out = self._cache.out
        return (MulOp.apply(MulOp.apply(out, MinusOp.apply(1, out)), grad_out)
                if out is not None else None)


class TanhOp(operator.Operator, fn_name="tanh", symbol="tanh"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.Variable):
        out = self._cache.out
        return (MulOp.apply(MinusOp(1, PowOp.apply(out, 2), grad_out))
                if out is not None else None)
