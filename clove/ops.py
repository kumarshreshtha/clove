from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple, Union

from clove import operator
from clove import variable

# TODO: throw an error when shapes don't match for the trailing dims.


def resolve_dims_for_reduction(dims, total_dims):
    if isinstance(dims, int):
        return resolve_dims_for_reduction((dims, ), total_dims)
    if dims is None:
        return tuple(range(total_dims))
    return tuple(total_dims + d if d < 0 else d for d in dims)


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
    old_shape = [1] * leading_dims + old_shape
    dims = []
    reduction_dims = []
    for i, (n_s, o_s) in enumerate(zip(old_shape, new_shape)):
        if n_s != o_s and (o_s != 1 or n_s != -1):
            raise RuntimeError(
                f"cannot expand dimension {i} of size {o_s} to size {n_s}")
        if o_s == 1 and n_s != 1:
            reduction_dims.append(i)
        dims.append(o_s if n_s == -1 else n_s)
    return dims, reduction_dims


class ExpandOp(operator.Operator, fn_name="expand"):
    def forward(self,
                x: variable.Variable,
                shape: Union[int, Tuple[int, ...]]):
        shape, self._cache.reduction_dims = (
            resolve_shape_for_expansion(shape, x.shape))
        return self.evaluate(x, shape)

    def backward(self, grad_out):
        return grad_out.sum(dim=self._cache.reduction_dim)


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
        return grad_output.reshape(self._cache.shape)


def expand_dims(shape, dims):
    return [s if i not in dims else 1 for i, s in enumerate(shape)]


class ReductionOp(operator.Operator):
    def forward(self,
                x: variable.Variable,
                dim: Union[int, Tuple[int, ...], None] = None,
                keepdim: bool = False
                ) -> variable.Variable:
        dim = resolve_dims_for_reduction(dim, len(x.shape))
        self._cache.dim = dim
        self._cache.keepdim = keepdim
        self._cache.shape = x.shape
        return self.evaluate(x, dim, keepdim)

    def backward(self, grad_out) -> variable.Variable:
        if not self._cache.keepdim:
            expanded_shape = expand_dims(self._cache.shape, self._cache.dim)
            grad_out = grad_out.reshape(expanded_shape)
        return grad_out


class SumOp(ReductionOp, fn_name="sum"):
    def backward(self, grad_out):
        return super().backward(grad_out).expand(self._cache.shape)


class MeanOp(ReductionOp, fn_name="mean"):
    def forward(self,
                x: variable.Variable,
                dim: Union[int, Tuple[int, ...], None] = None,
                keepdim: bool = False
                ) -> variable.Variable:
        self._cache.div = 1 / math.prod([x.shape[d] for d in dim])
        return super().forward(x, dim, keepdim)

    def backward(self, grad_out):
        grad_out = super().backward(grad_out) * self._cache.div
        return grad_out.expand(self._cache.shape)


class ProdOp(ReductionOp, fn_name="prod"):
    def forward(self,
                x: variable.Variable,
                dim: Union[int, Tuple[int, ...], None] = None,
                keepdim: bool = False
                ) -> variable.Variable:
        self._cache.x = x
        return super().forward(x, dim, keepdim)

    def backward(self, grad_out):
        grad_out = super().backward(grad_out)
        out = self._cache.out
        x = self._cache.x
        if not self._cache.keepdim:
            expanded_shape = expand_dims(x.shape, self._cache.dim)
            out = out.reshape(expanded_shape)
        grad = out / x
        return grad_out * grad


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
        return grad_out.transpose(self._cache.d1, self._cache.d0)


class PermuteOp(operator.Operator, fn_name="permute"):

    def forward(self, x: variable.ArrayOrScalar, dim: Sequence[int]):
        if self.requires_grad:
            self._cache.rev_dim = sorted(range(len(dim)), key=dim.__getitem__)
        return self.evaluate(x, dim)

    def backward(self, grad_out: variable.Variable):
        return grad_out.permute(self._cache.rev_dim)


class BinaryOp(operator.Operator):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        if (isinstance(x1, variable.Variable) and
            isinstance(x2, variable.Variable) and
                x1.shape != x2.shape):
            common_shape, x1_reduction_dim, x2_reduction_dim = (
                broadcast_shapes(x1, x2))
            x1 = x1.expand(common_shape) if x1.shape != common_shape else x1
            x2 = x2.expand(common_shape) if x2.shape != common_shape else x2
            if x1_reduction_dim:
                self._cache.x1_reduction_dim = x1_reduction_dim
            if x2_reduction_dim:
                self._cache.x2_reduction_dim = x2_reduction_dim
        return self.evaluate(x1, x2)

    def reduce_grad(self, x1_grad, x2_grad):
        if 'x1_reduction_dim' in self._cache and x1_grad is not None:
            x1_grad = x1_grad.sum(self._cache.x1_reduction_dim)
        if 'x2_reduction_dim' in self._cache and x2_grad is not None:
            x2_grad = x2_grad.sum(self._cache.x2_reduction_dim)
        return x1_grad, x2_grad


class AddOp(BinaryOp, fn_name="add", symbol="+"):
    def backward(self, grad_out: Optional[variable.ArrayOrScalar]):
        return self.reduce_grad(grad_out, grad_out)


class MulOp(operator.Operator, fn_name="multiply", symbol="<&times;>"):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return super().forward(x1, x2)

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        x1_grad = x2 * grad_out if x2 is not None else None
        x2_grad = x1 * grad_out if x1 is not None else None
        return self.reduce_grad(x1_grad, x2_grad)


class MinusOp(BinaryOp, fn_name="subtract", symbol="-"):
    def backward(self, grad_out: variable.Variable):
        return self.reduce_grad(grad_out, -grad_out)


class DivOp(BinaryOp, fn_name="divide"):
    def forward(self, x1: variable.ArrayOrScalar, x2: variable.ArrayOrScalar):
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        self._cache.x2 = (x2 if operator.prop_grad(x1) or
                          operator.prop_grad(x2)
                          else None)
        return super().forward(x1, x2)

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = grad_x2 = None
        if x2 is not None:
            grad_x1 = grad_out * x2.reciprocal()
        if x1 is not None:
            grad_x2 = grad_out * x1 * x2.reciprocal()**2
        return self.reduce_grad(grad_x1, grad_x2)

# TODO: do the following 2 ops qualify as binary ops?


class MatmulOp(operator.Operator, fn_name="matmul", symbol="@"):
    def forward(self, x1: variable.Variable, x2: variable.Variable):
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.Variable):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = grad_out @ x2.T if x2 is not None else None
        grad_x2 = x1.T @ grad_out if x1 is not None else None
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
            x1_grad = grad_out * x2 * x1 ** (x2 - 1)
        if out is not None:
            x2_grad = grad_out * x1.log()
        return x1_grad, x2_grad


class ReciprocalOp(operator.Operator, fn_name="reciprocal"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.Variable):
        return -grad_out * self._cache.out**2


class NegOp(operator.Operator, fn_name="negative", symbol="-1*"):
    def forward(self, x: variable.ArrayOrScalar):
        return self.evaluate(x)

    def backward(self, grad_out):
        return -grad_out


class ExpOp(operator.Operator, fn_name="exp", symbol="exp"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out):
        return grad_out * self._cache.out


class LogOp(operator.Operator, fn_name="log", symbol="ln"):
    def forward(self, x: variable.ArrayOrScalar):
        self._cache.x = x
        return self.evaluate(x)

    def backward(self, grad_out: variable.Variable):
        return grad_out * self._cache.x.reciprocal()


class SigmoidOp(operator.Operator, fn_name="sigmoid", symbol="<&sigma;>"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.Variable):
        out = self._cache.out
        return grad_out * out * (1 - out)


class TanhOp(operator.Operator, fn_name="tanh", symbol="tanh"):
    def forward(self, x: variable.ArrayOrScalar):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.Variable):
        return grad_out * (1 - self._cache.out**2)
