from __future__ import annotations

import functools
from typing import Sequence, Union
import numpy as np

from clove import backend
from clove import ops
from clove import variable


def get_data(array: variable.ArrayLike):
    if isinstance(array, variable.Variable):
        return array.data
    return array


def sigmoid(x: variable.ArrayLike):
    return np.reciprocal(1 + np.exp(-get_data(x)))


def transpose(x: variable.ArrayLike, dim0: int, dim1: int):
    return np.transpose(get_data(x), axes=(dim0, dim1))


def _resolve_unary(fn):
    @functools.wraps(fn)
    def wrapper(x: variable.ArrayLike,
                dim: Union[int, Sequence[int], None] = None):
        return fn(get_data(x)) if dim is None else fn(get_data(x), dim)
    return wrapper


def _resolve_binary(fn):
    @functools.wraps(fn)
    def wrapper(x1: variable.ArrayLike,
                x2: variable.ArrayLike):
        return fn(get_data(x1), get_data(x2))
    return wrapper


class Numpy(backend.Backend, name="numpy"):
    _OPS = {ops.CloneOp: _resolve_unary(np.copy),
            ops.AddOp: _resolve_binary(np.add),
            ops.ExpOp: _resolve_unary(np.exp),
            ops.ExpandOp: _resolve_unary(np.broadcast_to),
            ops.LogOp: _resolve_unary(np.log),
            ops.MatmulOp: _resolve_binary(np.matmul),
            ops.MulOp: _resolve_binary(np.multiply),
            ops.NegOp: _resolve_unary(np.negative),
            ops.PowOp: _resolve_binary(np.power),
            ops.SigmoidOp: sigmoid,
            ops.MinusOp: _resolve_binary(np.subtract),
            ops.SumOp: _resolve_unary(np.sum),
            ops.TanhOp: _resolve_unary(np.tanh),
            ops.TransposeOp: transpose,
            ops.PermuteOp: _resolve_unary(np.transpose)
            }

    @classmethod
    def resolve(cls, op, *args, **kwargs):
        return cls._OPS[op](*args, **kwargs)

    @classmethod
    def resolve_shape(cls, data):
        return data.shape

    @classmethod
    def resolve_dtype(cls, data):
        return data.dtype

    @classmethod
    def resolve_numel(cls, data):
        return data.size

    @classmethod
    def make_from_number(cls, data):
        return np.array(data)
