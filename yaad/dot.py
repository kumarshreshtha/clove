from typing import Dict
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

OP = dict(NODE_ATTR,
          shape="circle",
          fillcolor="dimgray",
          fontcolor="white",
          width="0.5",
          hieght="0.5",
          fixedsize="true")
OP_CACHE = dict(NODE_ATTR, shape="box", fillcolor="orange")
LEAF = dict(NODE_ATTR, shape="box", fillcolor="lightblue")
HIDDEN = dict(NODE_ATTR, shape="box", fillcolor="lightpink")
OUT = dict(NODE_ATTR, shape="box", fillcolor="darkolivegreen1")


def id_repr(data):
    return str(id(data))


def make_data_node(dot, data: node.Node, is_leaf: bool, show_grad: bool):
    if not show_grad or data.grad is None:
        dot.node(id_repr(data), str(data), **(LEAF if is_leaf else HIDDEN))
        return
    data_repr = str(data)
    grad_repr = str(data.grad)
    width = max(len(data_repr), len(grad_repr))
    n_repr = f"data : {data_repr.ljust(width)}\ngrad : {grad_repr.ljust(width)}"
    dot.node(id_repr(data), n_repr, **(LEAF if is_leaf else HIDDEN))


def add_cache_to_op(dot, op: ops.Operator):
    cache = op._cache
    if not cache:
        return
    max_key_len = len(max(cache.keys(), key=lambda x: len(x)))
    max_val_len = len(str(max(cache.values(), key=lambda x: len(str(x)))))
    cache_repr = "\n".join(
        [f"{k.ljust(max_key_len)} : {str(v).rjust(max_val_len)}"
         for k, v in cache.items()])
    dot.node(id_repr(cache), cache_repr, **OP_CACHE)
    dot.edge(id_repr(op), id_repr(op._cache), dir="none")


def make_out_node(dot, out: node.Node):
    dot.node(id_repr(out), str(out), **OUT)


def make_op_node(dot, op: ops.Operator):
    dot.node(id_repr(op), op.symbol, **OP)


def make_nodes(dot: graphviz.Digraph,
               root_op: ops.Operator,
               *,
               show_intermediate_outs: bool,
               show_saved: bool,
               show_grads: bool,
               _is_out=True,
               _visited=None,
               ):
    _visited = set() if _visited is None else _visited
    if root_op in _visited:
        return
    _visited.add(root_op)
    make_op_node(dot, root_op)
    for child in root_op.next_ops:
        if child is None:
            continue
        if child not in _visited:
            make_op_node(dot, child)
        dot.edge(id_repr(child), id_repr(root_op))
        make_nodes(dot,
                   child,
                   show_intermediate_outs=show_intermediate_outs,
                   show_saved=show_saved,
                   show_grads=show_grads,
                   _is_out=False,
                   _visited=_visited)
    if show_saved:
        add_cache_to_op(dot, root_op)
    if not _is_out and root_op.variable is not None:
        is_leaf = isinstance(root_op, ops.LeafOp)
        if not show_intermediate_outs and not is_leaf:
            return
        make_data_node(dot,
                       root_op.variable,
                       is_leaf=is_leaf,
                       show_grad=show_grads)
        dot.edge(id_repr(root_op.variable), id_repr(root_op), dir="none")


def make_dot(root: node.Node,
             *,
             show_intermediate_outs: bool = False,
             show_saved=False,
             show_grads=False):
    dot = graphviz.Digraph(node_attr=NODE_ATTR, graph_attr=dict(size="12,12"))
    make_out_node(dot, root)
    make_nodes(
        dot,
        root.op,
        show_intermediate_outs=show_intermediate_outs,
        show_saved=show_saved,
        show_grads=show_grads)
    dot.edge(id_repr(root.op), id_repr(root))
    return dot
