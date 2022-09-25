import collections
import enum
import functools
import inspect


class FunctionNames(str, enum.Enum):
    ADD = "add"
    MULTIPLY = "multiply"
    NEGATE = "neg"
    SUBTRACT = "sub"
    CLONE = "clone"
    EXP = "exp"
    POW = "pow"
    SIGMOID = "sigmoid"
    TANH = "tanh"


class _FunctionTable(collections.abc.MutableMapping):
    def __init__(self):
        self.__fn_names = frozenset(fn for fn in FunctionNames)
        self.__associations = dict()

    def __setitem__(self, key, value) -> None:
        if key not in self.__fn_names:
            raise KeyError()
        self.__associations[key] = value

    def __getitem__(self, key):
        return self.__associations[key]

    def __delitem__(self, key):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.__associations)

    def __len__(self):
        return len(self.__associations)


fn_table = _FunctionTable()


def register_operator(name, op):
    fn_table[name] = op


def walk_registry():
    yield from fn_table.items()


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


def make_fn(name, op):
    new_fn = _copy_op(name, op)
    sig = inspect.signature(op.forward)
    new_sig = sig.replace(
        parameters=[param for param in sig.parameters.values()
                    if param.name != "self"])
    new_fn.__signature__ = new_sig
    return new_fn


def make_method(name, op):
    new_fn = _copy_op(name, op)
    sig = inspect.signature(op.forward)
    new_sig = sig.replace(
        parameters=[param for param in sig.parameters.values()
                    if param.name != "input"])
    new_fn.__signature__ = new_sig
    return new_fn
