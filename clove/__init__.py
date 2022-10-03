from . import _math_ops
from .grad_mode import set_grad_enabled, is_grad_enabled, no_grad
from .variable import Variable
from . import autodiff
from . import dot
from . import _registry
from . import _backend
from . import make_bindings

# TODO: define set_backend at this level. it will remove all the creation
# ops from globals and replace them with the other ones.

for __name, __op in _registry.walk_registry():
    globals()[__name] = make_bindings.make_fn(__name, __op)

for __method_name, __fn_name in Variable.FUNCTION_ASSOCIATIONS.items():
    if __fn_name in _registry.fn_table:
        setattr(Variable,
                __method_name,
                make_bindings.make_method(__method_name,
                                          _registry.fn_table[__fn_name]))

__creation_ops = make_bindings.make_creation_ops()


def get_backend():
    return _backend.get_backend().name


for __op_name, __op in __creation_ops[get_backend()].items():
    globals()[__op_name] = __op


def set_backend(backend, /):
    if not _backend.has_backend(backend):
        raise ValueError(f"backend {backend} not found.")
    current_backend = _backend.get_backend()
    if current_backend.name == backend:
        return
    for op_name in __creation_ops[current_backend]:
        globals().pop(op_name, None)
    _backend.set_backend(backend)
    for op_name, op in __creation_ops[backend].items():
        globals()[op_name] = op
