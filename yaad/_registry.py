import enum
import functools
import types


class FunctionTable(enum.Enum, str):
    ADD = "add"
    MULTIPLY = "multiply"
    ...


def _placeholder_factory():
    def placeholder():
        raise NotImplementedError(
            f"{placeholder.__name__} has not been implemented yet.")
    return types.FunctionType(placeholder.__code__, globals={})


functional_registry = {}
for fn_name in FunctionTable:
    fn = _placeholder_factory()
    fn.__name__ = fn_name
    functional_registry[fn_name] = fn


def _wrap_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def register_op(name, op):
    placeholder = functional_registry[name]
    placeholder.__code__
