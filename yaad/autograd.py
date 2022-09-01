
from __future__ import annotations
from typing import List, Optional, Sequence

from yaad import ops
from yaad import node


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


def backward(output: node.Node,
             grad_output=None,
             retain_graph=False,
             create_graph=False):
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
    output = op.variable
    grad_output = op.grad_store.value
    op.grad_store.reset()
    if (accumulate_grad
            and output is not None
            and (output.is_leaf() or output.retains_grad)):
        output.grad = (grad_output if output.grad is None
                       else output.grad + grad_output)
    grads = op.backward(grad_output)
    for child, grad in zip(op.next_ops, grads):
        if child is not None:
            child.grad_store.update(grad)
    if not retain_graph:
        op.delete_edges()
    autodiff(ordered_ops, retain_graph, accumulate_grad)
