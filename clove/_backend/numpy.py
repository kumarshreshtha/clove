from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Sequence, Union
import numpy as np

from clove import ops
from clove._backend import base

if TYPE_CHECKING:
    from clove import variable


def sigmoid(x: np.ndarray):
    return np.reciprocal(1 + np.exp(-x))


def _resolve_unary(fn):
    @functools.wraps
    def wrapper(x: variable.Variable,
                dim: Union[int, Sequence[int], None] = None):
        return fn(x.data) if dim is None else fn(x.data, axis=dim)
    return wrapper


def _resolve_binary(fn):
    @functools.wraps
    def wrapper(x1: variable.Variable, x2: variable.Variable):
        return fn(x1.data, x2.data)
    return wrapper


def transpose(x: variable.Variable, dim0: int, dim1: int):
    return np.transpose(x.data, axes=(dim0, dim1))


@functools.lru_cache
def fn_associations():
    return {ops.CloneOp: _resolve_unary(np.copy),
            ops.AddOp: _resolve_binary(np.add),
            ops.ExpOp: _resolve_unary(np.exp),
            ops.LogOp: _resolve_unary(np.log),
            ops.MatmulOp: _resolve_binary(np.matmul),
            ops.MulOp: _resolve_binary(np.multiply),
            ops.NegOp: _resolve_unary(np.negative),
            ops.PowOp: _resolve_binary(np.power),
            ops.SigmoidOp: _resolve_unary(sigmoid),
            ops.MinusOp: _resolve_binary(np.subtract),
            ops.TanhOp: _resolve_unary(np.tanh),
            ops.TransposeOp: transpose,
            ops.PermuteOp: _resolve_unary(np.transpose)
            }


class Numpy(base.Backend, name="numpy"):

    _CREATION_ROUTINES = (
        np.empty,
        np.empty_like,
        np.eye,
        np.identity,
        np.ones,
        np.ones_like,
        np.zeros,
        np.zeros_like,
        np.full,
        np.full_like,
        np.array,
        np.asarray,
        np.asanyarray,
        np.ascontiguousarray,
        np.frombuffer,
        np.fromfile,
        np.fromfunction,
        np.fromiter,
        np.arange,
        np.linspace,
        np.logspace,
        np.geomspace,
        np.meshgrid,
        np.diag,
        np.diagflat,
        np.tri,
        np.tril,
        np.triu,
        np.vander
    )

    @classmethod
    def creation_routines(cls):
        return cls._CREATION_ROUTINES

    @classmethod
    def resolve(cls, op, *args, **kwargs):
        return fn_associations()[op](*args, **kwargs)
