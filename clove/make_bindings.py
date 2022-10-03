import collections
import functools
import inspect
import types
from typing import Optional

from clove import _backend
from clove import operator
from clove import variable


def _bind_free_vars(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def _copy_op(name, op: operator.Operator):
    new_fn = _bind_free_vars(op.apply)
    new_fn.__doc__ = op.forward.__doc__
    new_fn.__annotations__ = op.forward.__annotations__
    new_fn.__defaults__ = op.forward.__defaults__
    new_fn.__name__ = name
    return new_fn


def make_fn(name, op: operator.Operator):
    new_fn = _copy_op(name, op)
    sig = inspect.signature(op.forward)
    new_sig = sig.replace(
        parameters=[param for param in sig.parameters.values()
                    if param.name != "self"])
    new_fn.__signature__ = new_sig
    return new_fn


def make_method(name, op: operator.Operator):
    new_fn = _copy_op(name, op)
    sig = inspect.signature(op.forward)
    new_sig = sig.replace(
        parameters=[param for param in sig.parameters.values()
                    if param.name != "input"])
    new_fn.__signature__ = new_sig
    return new_fn


def update_signature(sig: inspect.Signature) -> inspect.Signature:
    req_grad = inspect.Parameter(
        name="requires_grad",
        kind=inspect.Parameter.KEYWORD_ONLY,
        annotation=bool,
        default=False)
    name = inspect.Parameter(
        name="name",
        kind=inspect.Parameter.KEYWORD_ONLY,
        annotation=Optional[str],
        default=None,
    )
    parameters = (
        [param for param in sig.parameters.values()] + [req_grad, name])
    return sig.replace(parameters=parameters,
                       return_annotation=variable.Variable)


def creation_op_wrapper(fn):
    @functools.wraps(fn)
    def runner(*args, requires_grad=False, name=None, **kwargs):
        data = fn(*args, **kwargs)
        return variable.Variable(data, requires_grad=requires_grad, name=name)

    if not isinstance(fn, types.BuiltinFunctionType):
        try:
            sig = update_signature(inspect.signature(fn))
            runner.__signature__ = sig
            runner.__defaults__ = tuple(
                param.default for param in sig.parameters.values()
                if param.default is not inspect.Parameter.empty)
            runner.__annotations__ = {param.name: param.annotation
                                      for param in sig.parameters.values()}
        except ValueError:
            return runner
    return runner


def make_creation_ops():
    ops = collections.defaultdict(dict)
    for bk_name, bk in _backend.backends():
        bk: _backend.Backend
        for op in bk.creation_routines():
            ops[bk_name][op.__name__] = creation_op_wrapper(op)
    return ops


# def bind_fn_association():
#     ops = collections.defaultdict(dict)
#     for bk_name, bk in backend.backends():
#         bk: backend.Backend
#         for op in bk.fn_associations():
#             ops[bk_name][op.__name__] = make_creation_op(op)
#     return ops
