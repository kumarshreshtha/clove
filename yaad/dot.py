import graphviz

from yaad import node
from yaad import ops

NODE_ATTR = dict(align="left",
                 fontsize="10",
                 ranksep="0.1",
                 height="0.2",
                 fontname="monospace",
                 style="filled",
                 color="black")

OP = dict(NODE_ATTR, shape="circle", fillcolor="dimgray", fontcolor="white")
OP_CACHE = dict(NODE_ATTR, shape="record", fillcolor="orange")
LEAF = dict(NODE_ATTR, shape="record", fillcolor="lightblue")
HIDDEN = dict(NODE_ATTR, shape="record", fillcolor="lightpink")
OUT = dict(NODE_ATTR, shape="record", fillcolor="darkolivegreen1")


# TODO: improve graph representations, the leaves looks really big.
# maybe avoid leaves altogether?
# maybe repr op name rather than symbol?
# remove name field when the name is None.

def id_repr(data):
    return str(id(data))


def make_data_node(dot, data: node.Node, is_leaf: bool):
    dot.node(id_repr(data),
             f"name:{data.name}|data:{data.data}|grad:{data.grad}",
             **(LEAF if is_leaf else HIDDEN))


def make_out_node(dot, out: node.Node):
    dot.node(id_repr(out), f"name:{out.name}|data:{out.data}", **OUT)


def make_op_node(dot, op: ops.Operator):
    dot.node(id_repr(op), op.symbol, **OP)

# TODO: implement show cache feature later.


def make_nodes(dot: graphviz.Digraph, root_op: ops.Operator, _visited=None):
    _visited = set() if _visited is None else _visited
    if root_op in _visited:
        return
    make_op_node(dot, root_op)
    for child in root_op.next_ops:
        if child is None:
            continue
        if child not in _visited:
            make_op_node(dot, child)
        dot.edge(id_repr(child), id_repr(root_op))
        make_nodes(dot, child)
    if isinstance(root_op, ops.LeafOp):
        make_data_node(dot, root_op.variable, is_leaf=True)
        dot.edge(id_repr(root_op.variable), id_repr(root_op))


def make_dot(root: node.Node):
    dot = graphviz.Digraph(node_attr=NODE_ATTR, graph_attr=dict(size="12,12"))
    make_out_node(dot, root)
    make_nodes(dot, root.op)
    dot.edge(id_repr(root.op), id_repr(root))
    return dot
