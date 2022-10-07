from __future__ import annotations

import collections
import dataclasses
import enum
import inspect
from typing import Sequence, Union

# should we define dtypes? will need associations for that too.


@dataclasses.dataclass
class OperatorDefinition:
    name: str
    signature: inspect.Signature


def make_unary_signature(on_axes=False):
    data = inspect.Parameter(name='x',
                             kind=inspect.Parameter.POSITIONAL_ONLY,
                             annotation="variable.Variable")
    params = [data]
    if on_axes:
        axis = inspect.Parameter(name='dim',
                                 kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                 annotation=Union[int, Sequence[int]])
        params.append(axis)
    return_annotation = "variable.Variable"
    return inspect.Signature(params, return_annotation)


def make_binary_signature(allow_number=False):
    data1 = inspect.Parameter(name='x1',
                              kind=inspect.Parameter.POSITIONAL_ONLY,
                              annotation="variable.Variable")
    data2_annotation = ("Union['variable.Variable', numbers.Number]"
                        if allow_number else "variable.Variable")
    data2 = inspect.Parameter(name='x2',
                              kind=inspect.Parameter.POSITIONAL_ONLY,
                              annotation=data2_annotation)
    return_annotation = "variable.Variable"
    return inspect.Signature([data1, data2], return_annotation)


class Function(OperatorDefinition, enum.Enum):
    ADD = OperatorDefinition(name="add",
                             signature=make_binary_signature(True))
    MULTIPLY = OperatorDefinition(name="multiply",
                                  signature=make_binary_signature(True))
    SUBTRACT = OperatorDefinition(name="add",
                                  signature=make_binary_signature(True))
    DIVIDE = OperatorDefinition(name="divide",
                                signature=make_binary_signature(True))
    POW = OperatorDefinition(name="power",
                             signature=make_binary_signature(True))
    MATMUL = OperatorDefinition(name="divide",
                                signature=make_binary_signature())
    NEGATIVE = OperatorDefinition(name="negative",
                                  signature=make_unary_signature())
    EXP = OperatorDefinition(name="exp",
                             signature=make_unary_signature())
    SIGMOID = OperatorDefinition(name="sigmoid",
                                 signature=make_unary_signature())
    TANH = OperatorDefinition(name="tanh",
                              signature=make_unary_signature())
    LOG = OperatorDefinition(name="tanh",
                             signature=make_unary_signature())
    CLONE = OperatorDefinition(name="clone",
                               signature=make_unary_signature())


class CreationRoutines:
    CLONE = OperatorDefinition(name="clone",
                               signature=make_unary_signature())


class ManipulationRoutines:
    TRANSPOSE = OperatorDefinition(name="transpose",
                                   signature=make_unary_signature(True))


# class _FunctionTable(collections.abc.MutableMapping):
#     def __init__(self):
#         self.__fn_names = frozenset(fn for fn in Function)
#         self.__associations = dict()

#     def __setitem__(self, key, value) -> None:
#         if key not in self.__fn_names:
#             raise KeyError()
#         self.__associations[key] = value

#     def __getitem__(self, key):
#         return self.__associations[key]

#     def __delitem__(self, key):
#         raise NotImplementedError

#     def __iter__(self):
#         return iter(self.__associations)

#     def __len__(self):
#         return len(self.__associations)


# fn_table = _FunctionTable()


# def register_operator(name, op):
#     fn_table[name] = op


# def walk_registry():
#     yield from fn_table.items()
