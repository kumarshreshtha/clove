from .grad_mode import set_grad_enabled, is_grad_enabled, no_grad
from .variable import Variable
from . import autodiff
from . import dot
from . import _registry

for __name, __fn in _registry.walk_registry():
    globals()[__name] = __fn

for __method_name, __fn_name in Variable.FUNCTION_ASSOCIATIONS:
    setattr(Variable, __method_name, _registry.fn_table[__fn_name])
