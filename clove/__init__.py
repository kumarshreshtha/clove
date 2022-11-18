from . import autodiff
from ._creation_routines import *
from ._op_bindings import *
from .backend import get_backend, set_backend
from .backends import cupy, numpy
from .dot import make_dot
from .grad_mode import set_grad_enabled, is_grad_enabled, no_grad

# TODO: add a check to ensure the backend implements all ops
# TODO: need to unify backend design.
set_backend('numpy')  # default to numpy
