from __future__ import annotations

import functools
import inspect
import pathlib
import textwrap
import types
from typing import TYPE_CHECKING, Optional

from clove import operator
from clove import variable

if TYPE_CHECKING:
    from clove import backend


def get_op_signature(cls: operator.Operator):
    signature = inspect.signature(cls.forward)
    return signature.replace(
        parameters=[param for param in signature.parameters.values()
                    if param.name != "self"])


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


def make_fn(op: operator.Operator):
    new_fn = _copy_op(op.fn_name, op)
    new_fn.__signature__ = get_op_signature(op)
    new_fn._op = op
    return new_fn


def make_method(name, op: operator.Operator):
    new_fn = _copy_op(name, op)
    sig = get_op_signature(op)
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


def wrap_creation_op(fn, bk: backend.Backend):
    @functools.wraps(fn)
    def wrapper(*args,
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
            wrapper.__signature__ = sig
            wrapper.__defaults__ = tuple(
                param.default for param in sig.parameters.values()
                if param.default is not inspect.Parameter.empty)
            wrapper.__annotations__ = {param.name: param.annotation
                                       for param in sig.parameters.values()}
    new_doc = f"""
    {wrapper.__name__}(*args, requires_grad=False, name=None, **kwargs)

    clove args:

        requires_grad (bool) : When true, the variable is marked as
            differentiable and it's computation graphs are tracked.

        name (Optional[str]) : Optional name for the variable. The variable
            is identified with this name in computation graph visualizations
            with `clove.make_dot()`.

    Backend Docs:

    {wrapper.__doc__}
    """
    wrapper.__doc__ = new_doc

    return wrapper


def generate_static_op_bindings(filename="_op_bindings.py",
                                basedir=pathlib.Path(__file__).parent):

    code = textwrap.dedent("""
    from clove import binding_utils
    from clove import ops

    """)
    bindings = "\n".join([
        f"{name} = binding_utils.make_fn(ops.{op.__name__})"
        for name, op in operator.fn_registry.items()
    ])

    code = code + bindings
    basedir.mkdir(parents=True, exist_ok=True)
    with open(basedir / filename, "w") as f:
        f.write(code)
