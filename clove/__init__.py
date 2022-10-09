from .grad_mode import set_grad_enabled, is_grad_enabled, no_grad
# from . import variable
from . import autodiff
from . import dot
from . import backends
from .backend import get_backend, set_backend
from .backends import cupy
from .backends import numpy
from ._op_bindings import *

# TODO: add a check to ensure the backend implements all ops

set_backend('numpy')  # default to numpy
