import numpy as np

from clove import _registry
from clove._backend import backend

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


def sigmoid(x: np.ndarray):
    return np.reciprocal(1 + np.exp(-x))


_OPS = {
    _registry.Function.CLONE: np.copy,
    _registry.Function.ADD: np.add,
    _registry.Function.EXP: np.exp,
    _registry.Function.LOG: np.log,
    _registry.Function.MATMUL: np.matmul,
    _registry.Function.MULTIPLY: np.multiply,
    _registry.Function.NEGATE: np.negative,
    _registry.Function.POW: np.power,
    _registry.Function.SIGMOID: sigmoid,
    _registry.Function.SUBTRACT: np.subtract,
    _registry.Function.TANH: np.tanh,
    _registry.Function.TRANSPOSE: np.transpose
}


class Numpy(backend.Backend, name="numpy"):

    @classmethod
    def creation_routines(cls):
        return _CREATION_ROUTINES

    @classmethod
    def fn_associations(cls):
        return _OPS
