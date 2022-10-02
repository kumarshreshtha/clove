import collections
import enum

# TODO: make this Function template more powerful.
# We can define BinaryOps, UnaryOps etc to have definite signatures.
# Likely a dataclass with signature, name etc.
# then both backend fn and Op define associations to this signature for
# for proper sync. the routing can be controlled by either Operator or
# a middle man.


class Function(str, enum.Enum):
    ADD = "add"
    MULTIPLY = "multiply"
    NEGATE = "neg"
    SUBTRACT = "sub"
    CLONE = "clone"
    EXP = "exp"
    POW = "pow"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    MATMUL = "matmul"
    TRANSPOSE = "transpose"
    LOG = "log"


class _FunctionTable(collections.abc.MutableMapping):
    def __init__(self):
        self.__fn_names = frozenset(fn for fn in Function)
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
