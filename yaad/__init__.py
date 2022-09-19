from .grad_mode import set_grad_enabled, is_grad_enabled, no_grad
from .variable import Variable
from . import ops
from . import autodiff
from . import dot
from . import _registry

for __name, __op in _registry.walk_registry():
    globals()[__name] = _registry.make_fn(__name, __op)

for __method_name, __fn_name in Variable.FUNCTION_ASSOCIATIONS.items():
    setattr(
        Variable,
        __method_name,
        _registry.make_method(__method_name, _registry.fn_table[__fn_name]))
