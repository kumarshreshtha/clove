import enum
import functools
import inspect


class FunctionNames(str, enum.Enum):
    ADD = "add"
    MULTIPLY = "multiply"
    SUBTRACT = "sub"
    CLONE = "clone"
    EXP = "exp"
    POW = "pow"
    SIGMOID = "sigmoid"

# TODO: convert this to MutableMapping


class _FunctionTable:
    def __init__(self):
        self._fn_names = frozenset(fn for fn in FunctionNames)
        self._associations = dict()

    def __setitem__(self, key, value) -> None:
        if key not in self._fn_names:
            raise KeyError()
        self._associations[key] = value

    def __getitem__(self, key):
        return self._associations[key]


def _bind_free_vars(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def _copy_op(name, op):
    new_fn = _bind_free_vars(op.apply)
    new_fn.__doc__ = op.forward.__doc__
    new_fn.__annotations__ = op.forward.__annotations__
    new_fn.__defaults__ = op.forward.__defaults__
    new_fn.__name__ = name
    return new_fn


def _op_to_functional(name, op):
    new_fn = _copy_op(name, op)
    sig = inspect.signature(op.forward)
    new_sig = sig.replace(
        parameters=[param for param in sig.parameters.values()
                    if param.name != "self"])
    new_fn.__signature__ = new_sig
    return new_fn


def register_op_method(name, op):
    new_fn = _copy_op(name, op)
    sig = inspect.signature(op.forward)
    new_sig = sig.replace(
        parameters=[param for param in sig.parameters.values()
                    if param.name != "input"])
    new_fn.__signature__ = new_sig


fn_table = _FunctionTable()


def register_functional(name, op):
    fn_table[name] = _op_to_functional(name, op)


def walk_registry():
    yield from fn_table._associations.items()
