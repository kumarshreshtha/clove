from __future__ import annotations

import functools
from typing import Sequence, Union
import numpy as np

from clove import backend
from clove import ops
from clove import variable


# TODO: move data abstraction to operator function so direct function
# associations can be created whenever possible.

# TODO: need to refactor resolution into groups, manipulation routines,
# index routines, unary on dim, unary, binary etc.

def sigmoid(x: np.ndarray):
    return np.reciprocal(1 + np.exp(-x))


def transpose(x: np.ndarray, dim0: int, dim1: int):
    return np.transpose(x, axes=(dim0, dim1))


def index(x: variable.Variable, key):
    return x.data[key]


def _resolve_unary_on_dim(fn):
    @functools.wraps(fn)
    def wrapper(x: np.ndarray,
                dim: Union[int, Sequence[int], None] = None,
                keepdim: bool = False):
        return fn(x, dim, keepdims=keepdim)
    return wrapper


def _resolve_unary(fn):
    @functools.wraps(fn)
    def wrapper(x: np.ndarray,
                dim: Union[int, Sequence[int], None] = None):
        return fn(x) if dim is None else fn(x, dim)
    return wrapper


def _resolve_binary(fn):
    @functools.wraps(fn)
    def wrapper(x1: np.ndarray,
                x2: np.ndarray):
        return fn(x1, x2)
    return wrapper


def _create_from_shape(fn):
    @functools.wraps(fn)
    def wrapper(*shape):
        return fn(shape)
    return wrapper


class Numpy(backend.Backend, name="numpy"):
    _OPS = {ops.CloneOp: _resolve_unary(np.copy),
            ops.AddOp: _resolve_binary(np.add),
            ops.ExpOp: _resolve_unary(np.exp),
            ops.ExpandOp: _resolve_unary(np.broadcast_to),
            ops.LogOp: _resolve_unary(np.log),
            ops.MatmulOp: _resolve_binary(np.matmul),
            ops.MeanOp: _resolve_unary(np.mean),
            ops.MulOp: _resolve_binary(np.multiply),
            ops.DivOp: _resolve_binary(np.divide),
            ops.NegOp: _resolve_unary(np.negative),
            ops.PowOp: _resolve_binary(np.power),
            ops.SigmoidOp: sigmoid,
            ops.MinusOp: _resolve_binary(np.subtract),
            ops.SumOp: _resolve_unary_on_dim(np.sum),
            ops.TanhOp: _resolve_unary(np.tanh),
            ops.TransposeOp: transpose,
            ops.PermuteOp: _resolve_unary(np.transpose),
            ops.ReshapeOp: _resolve_unary(np.reshape),
            ops.IndexOp: index,
            ops.ProdOp: _resolve_unary_on_dim(np.prod),
            }

    _creation_routines = {
        'empty': _create_from_shape(np.empty),
        'ones': _create_from_shape(np.ones),
        'zeros': _create_from_shape(np.zeros),
        'array': np.array,
        'rand': _create_from_shape(np.random.rand),
        'randn': _create_from_shape(np.random.randn),
        'randint': np.random.randint
    }

    @classmethod
    def resolve(cls, op):
        return cls._OPS[op]

    @classmethod
    def resolve_creation_routine(cls, fn_name):
        return cls._creation_routines[fn_name]

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
