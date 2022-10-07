import functools
import inspect
import numpy as np

from clove import ops
from clove._backend import backend


def sigmoid(x: np.ndarray):
    return np.reciprocal(1 + np.exp(-x))


def _resolve_unary(fn):
    @functools.wraps
    def wrapper(binding: inspect.BoundArguments):
        if 'dim' in binding.arguments:
            return fn(binding.arguments['x'], axis=binding.arguments['dim'])
        return fn(binding.arguments['x'])
    return wrapper


def _resolve_binary(fn):
    @functools.wraps
    def wrapper(binding: inspect.BoundArguments):
        return fn(binding.arguments['x1'], binding.arguments['x2'])
    return wrapper


def _resolve_transpose(fn):
    @functools.wraps
    def wrapper(binding: inspect.BoundArguments):
        axes = (binding.arguments['dim0'], binding.arguments['dim1'])
        return fn(binding.arguments['x'], axes=axes)
    return wrapper


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
            ops.TransposeOp: _resolve_transpose(np.transpose),
            ops.PermuteOp: _resolve_unary(np.transpose)
            }


class Numpy(backend.Backend, name="numpy"):

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
    def resolve(cls, op, binding):
        return fn_associations[op](binding)


# def resolve(fn:_registry.Function, binding):
#     if fn is _registry.
