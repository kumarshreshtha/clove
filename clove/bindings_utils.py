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
    new_fn.__doc__ = op.__doc__
    new_fn.__annotations__ = op.forward.__annotations__
    new_fn.__defaults__ = op.forward.__defaults__
    new_fn.__name__ = name
    return new_fn


def make_fn(name, op: operator.Operator):
    new_fn = _copy_op(name, op)
    new_fn.__signature__ = op.get_signature()
    return new_fn


def make_method(name, op: operator.Operator):
    new_fn = _copy_op(name, op)
    sig = op.get_signature()
    self_param = inspect.Parameter(
        "self", kind=inspect.Parameter.POSITIONAL_OR_KEYWORD)
    new_parameters = [self_param,
                      *[param for param in list(sig.parameters.values())[1:]]]
    new_sig = sig.replace(parameters=new_parameters)
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


def creation_op_wrapper(fn, bk):
    @functools.wraps(fn)
    def runner(*args,
               requires_grad: bool = False,
               name: Optional[str] = None,
               **kwargs):
        args = list(args)
        for i, arg in enumerate(args):
            if isinstance(arg, variable.Variable):
                args[i] = arg.data
        for k, v in kwargs.items():
            if isinstance(v, variable.Variable):
                kwargs[k] = v.data
        data = fn(*args, **kwargs)
        return variable.Variable(
            data, backend=bk, requires_grad=requires_grad, name=name)

    if not isinstance(fn, types.BuiltinFunctionType):
        try:
            sig = update_signature(inspect.signature(fn))
        except ValueError:
            sig = None
        if not sig is None:
            runner.__signature__ = sig
            runner.__defaults__ = tuple(
                param.default for param in sig.parameters.values()
                if param.default is not inspect.Parameter.empty)
            runner.__annotations__ = {param.name: param.annotation
                                      for param in sig.parameters.values()}
    doc = runner.__doc__
    new_doc = f"""
    {runner.__name__}(*args, requires_grad=False, name=None, **kwargs)

    clove args:

        requires_grad (bool) : When true, the variable is marked as
            differentiable and it's computation graphs are tracked.

        name (Optional[str]) : Optional name for the variable. The variable
            is identified with this name in computation graph visualizations
            with `clove.make_dot()`.

    Backend Docs:

    {doc}
    """

    runner.__doc__ = new_doc

    return runner


def make_creation_ops():
    ops = collections.defaultdict(dict)
    for bk_name, bk in _backend.backends():
        bk: _backend.Backend
        for op in bk.creation_routines():
            ops[bk_name][op.__name__] = creation_op_wrapper(op, bk)
    return ops
