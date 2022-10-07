import functools
import inspect
import numpy as np

from clove import definitions
from clove._backend import backend


def sigmoid(x: np.ndarray):
    return np.reciprocal(1 + np.exp(-x))


def _resolve_unary(fn):
    @functools.wraps
    def wrapper(binding: inspect.BoundArguments):
        if axis := 'dim' in binding.arguments:
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

    _OPS = {
        definitions.Function.CLONE: _resolve_unary(np.copy),
        definitions.Function.ADD: _resolve_binary(np.add),
        definitions.Function.EXP: _resolve_unary(np.exp),
        definitions.Function.LOG: _resolve_unary(np.log),
        definitions.Function.MATMUL: _resolve_binary(np.matmul),
        definitions.Function.MULTIPLY: _resolve_binary(np.multiply),
        definitions.Function.NEGATIVE: _resolve_unary(np.negative),
        definitions.Function.POW: _resolve_binary(np.power),
        definitions.Function.SIGMOID: _resolve_unary(sigmoid),
        definitions.Function.SUBTRACT: _resolve_binary(np.subtract),
        definitions.Function.TANH: _resolve_unary(np.tanh),
        definitions.Function.TRANSPOSE: _resolve_transpose(np.transpose),
        definitions.Function.PERMUTE: _resolve_unary(np.transpose)
    }

    @classmethod
    def creation_routines(cls):
        return cls._CREATION_ROUTINES

    def resolve(cls, defn, binding):
        return cls._OPS[defn](binding)


# def resolve(fn:_registry.Function, binding):
#     if fn is _registry.
