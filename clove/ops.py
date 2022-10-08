from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from clove import operator

if TYPE_CHECKING:
    from clove import variable


# TODO: add something to throw errors on backend mismatch.
# a variable will store it's creation backend.

# TODO: make it easy to add backends and ops from outside clove

class CloneOp(operator.Operator,
              symbol="clone"):
    def forward(self, x: variable.ArrayLike):
        return self.evaluate(x)

    def backward(self, grad_out: variable.ArrayLike):
        return grad_out


class TransposeOp(operator.Operator,
                  symbol="T"):

    def forward(self, x: variable.ArrayLike, dim_0: int, dim_1: int):
        self._cache.d0 = dim_0
        self._cache.d1 = dim_1
        return self.evaluate(x, dim_0, dim_1)

    def backward(self, grad_out: variable.ArrayLike):
        return TransposeOp.apply(grad_out, self._cache.d1, self._cache.d0)


class PermuteOp(operator.Operator):

    def forward(self,
                x: variable.ArrayLike,
                dim: Sequence[int],
                ):
        if self.requires_grad:
            self._cache.rev_dim = sorted(range(len(dim)), key=dim.__getitem__)
        return self.evaluate(x, dim)

    def backward(self, grad_out: variable.ArrayLike):
        return PermuteOp.apply(grad_out, self._cache.rev_dim)


class AddOp(operator.Operator,
            symbol="+"):
    def forward(self, x1: variable.ArrayLike, x2: variable.ArrayLike):
        return self.evaluate(x1, x2)

    def backward(self, grad_out: Optional[variable.ArrayLike]):
        return grad_out, grad_out


class MulOp(operator.Operator,
            symbol="<&times;>"):
    def forward(self, x1: variable.ArrayLike, x2: variable.ArrayLike):
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.ArrayLike):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = MulOp.apply(x2, grad_out) if x2 is not None else None
        grad_x2 = MulOp.apply(x1, grad_out) if x1 is not None else None
        return grad_x1, grad_x2


class MatmulOp(operator.Operator,
               symbol="@"):
    def forward(self, x1: variable.Variable, x2: variable.Variable):
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.x1 = x1 if operator.prop_grad(x2) else None
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.ArrayLike):
        x1, x2 = self._cache.x1, self._cache.x2
        grad_x1 = (MatmulOp.apply(grad_out, TransposeOp.apply(x2))
                   if x2 is not None else None)
        grad_x2 = (MatmulOp.apply(TransposeOp.apply(x1), grad_out)
                   if x1 is not None else None)
        return grad_x1, grad_x2


class NegOp(operator.Operator,
            symbol="-1*"):
    def forward(self, x: variable.ArrayLike):
        return self.evaluate(x)

    def backward(self, grad_out):
        return NegOp.apply(grad_out)


class MinusOp(operator.Operator,
              symbol="-"):
    def forward(self, x1: variable.ArrayLike, x2: variable.ArrayLike):
        return self.evaluate(x1, x2)

    def backward(self, grad_out: variable.ArrayLike):
        return grad_out, NegOp.apply(grad_out)


class ExpOp(operator.Operator,
            symbol="exp"):
    def forward(self, x: variable.ArrayLike):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out):
        out = self._cache.out
        return MulOp.apply(out, grad_out) if out is not None else None


class LogOp(operator.Operator,
            symbol="ln"):
    def forward(self, x: variable.ArrayLike):
        self._cache.x = x
        return self.evaluate(x)

    def backward(self, grad_out: variable.ArrayLike):
        x = self._cache.x
        return (MulOp.apply(PowOp.apply(x, -1), grad_out)
                if x is not None else None)


class PowOp(operator.Operator,
            symbol="**"):
    def forward(self, x1: variable.ArrayLike, x2: variable.ArrayLike):
        out = self.evaluate(x1, x2)
        self._cache.x1 = (
            x1 if operator.prop_grad(x1) or operator.prop_grad(x2) else None)
        self._cache.x2 = x2 if operator.prop_grad(x1) else None
        self._cache.out = out if operator.prop_grad(x2) else None
        return out

    def backward(self, grad_out: variable.ArrayLike):
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


class SigmoidOp(operator.Operator,
                symbol="<&sigma;>"):
    def forward(self, x: variable.ArrayLike):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.ArrayLike):
        out = self._cache.out
        return (MulOp.apply(MulOp.apply(out, MinusOp.apply(1, out)), grad_out)
                if out is not None else None)


class TanhOp(operator.Operator,
             symbol="tanh"):
    def forward(self, x: variable.ArrayLike):
        out = self.evaluate(x)
        self._cache.out = out
        return out

    def backward(self, grad_out: variable.ArrayLike):
        out = self._cache.out
        return (MulOp.apply(MinusOp(1, PowOp.apply(out, 2), grad_out))
                if out is not None else None)
