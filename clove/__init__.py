from .grad_mode import set_grad_enabled, is_grad_enabled, no_grad
from .variable import Variable
from . import ops
from . import autodiff
from . import dot
from . import bindings_utils
from . import backend
from . import backends

backend.Backend.set_backend(backends.Numpy)  # default to numpy

# for __name, __op in definitions.walk_registry():
#     globals()[__name] = make_bindings.make_fn(__name, __op)

# for __method_name, __fn_name in Variable.METHODS_FROM_DEFN.items():
#     if __fn_name in definitions.fn_table:
#         setattr(Variable,
#                 __method_name,
#                 make_bindings.make_method(__method_name,
#                                           definitions.fn_table[__fn_name]))

# Maybe just do it for the current backend?
__creation_ops = bindings_utils.make_creation_ops()

# for __op_name, __op in __creation_ops[get_backend()].items():
#     globals()[__op_name] = __op


def set_backend(name, /):
    if not backend.has_backend(name):
        raise ValueError(f"backend {name} not found.")
    current_backend = backend.get_backend()
    if current_backend.name == name:
        return
    for op_name in __creation_ops[current_backend]:
        globals().pop(op_name, None)
    backend.set_backend(name)
    for op_name, op in __creation_ops[name].items():
        globals()[op_name] = op
