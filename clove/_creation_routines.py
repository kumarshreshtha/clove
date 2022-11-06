import functools

from clove import backend as _backend
from clove import variable


def _creation_routine(fn):
    fn_name = fn.__name__

    @functools.wraps(fn)
    def wrapper(*args,
                requires_grad=False,
                name=None,
                **kwargs):
        _bk = _backend.get_backend()
        fn = _bk.resolve_creation_routine(fn_name)
        return variable.Variable(fn(*args, **kwargs),
                                 backend=_bk,
                                 requires_grad=requires_grad,
                                 name=name)
    return wrapper


@_creation_routine
def empty(*shape, requires_grad=False, name=None):
    ...


@_creation_routine
def empty_like(input, requires_grad=False, name=None):
    ...


@_creation_routine
def eye(n, m=None, requires_grad=False, name=None):
    ...


@_creation_routine
def identity(n, requires_grad=False, name=None):
    ...


@_creation_routine
def ones(*shape, requires_grad=False, name=None):
    ...


@_creation_routine
def ones_like(input, requires_grad=False, name=None):
    ...


@_creation_routine
def zeros(*shape, requires_grad=False, name=None):
    ...


@_creation_routine
def zeros_like(input, requires_grad=False, name=None):
    ...


@_creation_routine
def full(*shape, requires_grad=False, name=None):
    ...


@_creation_routine
def full_like(input, requires_grad=False, name=None):
    ...


@_creation_routine
def array(input, requires_grad=False, name=None):
    ...


@_creation_routine
def arange(start, end, step=1, requires_grad=False, name=None):
    ...


@_creation_routine
def meshgrid(*arrays, indexing='xy', requires_grad=False, name=None):
    ...


@_creation_routine
def rand(*shape, requires_grad=False, name=None):
    ...


@_creation_routine
def rand_like(input, requires_grad=False, name=None):
    ...


@_creation_routine
def randint(low, high, size, requires_grad=False, name=None):
    ...


@_creation_routine
def randint_like(input, requires_grad=False, name=None):
    ...


@_creation_routine
def randn(*shape, requires_grad=False, name=None):
    ...


@_creation_routine
def randn_like(input, requires_grad=False, name=None):
    ...


@_creation_routine
def normal(*shape, requires_grad=False, name=None):
    ...
