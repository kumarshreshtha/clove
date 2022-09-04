
from __future__ import annotations
from typing import List, Optional, Sequence

from yaad import ops
from yaad import node
from yaad import grad_mode


def topological_order(root: Optional[ops.Operator],
                      _order=None,
                      _visited=None) -> List[ops.Operator]:
    if root is None:
        return _order
    _visited = set() if _visited is None else _visited
    _order = [] if _order is None else _order
    if root in _visited:
        return _order
    _visited.add(root)
    for op in root._children:
        _order = topological_order(op, _order, _visited)
    _order.append(root)
    return _order

# TODO: remove create_graph from here and add it exclusively to grad (with
# default = True maybe?)

# TODO: What to do when there is no path from output to input? zero or none?
# torch throws an error.
# When you know the outputs and the inputs, you don't need to traverse the full
# graph? find all paths from out to in and mark nodes. then only traverse
# those nodes.


def backward(output: node.Node,
             grad_output=None,
             retain_graph=False,
             create_graph=False):
    with grad_mode.set_grad_enabled(create_graph):
        grad_output = (node.Node(1., requires_grad=create_graph)
                       if grad_output is None else grad_output)
        output.op.grad_store.update(grad_output)
        ordered_ops = topological_order(output.op)
        autodiff(ordered_ops, retain_graph, accumulate_grad=True)


def grad():
    ...


def autodiff(ordered_ops: List[ops.Operator],
             retain_graph: bool,
             accumulate_grad=True):
    op = ordered_ops.pop()
    grad_output = op.grad_store.value
    op.grad_store.reset()
    output = op.variable
    if (accumulate_grad
            and output is not None
            and (output.is_leaf() or output.retains_grad)):
        output.grad = (grad_output if output.grad is None
                       else output.grad + grad_output)
    grads = op.backward(grad_output)
    grads = (grads,) if isinstance(grads, node.Node) else grads
    for child, grad in zip(op.next_ops, grads):
        if child is not None:
            child.grad_store.update(grad)
    if not retain_graph:
        op.clear_cache()
    if ordered_ops:
        autodiff(ordered_ops, retain_graph, accumulate_grad)
