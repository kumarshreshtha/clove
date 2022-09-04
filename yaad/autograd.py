
from __future__ import annotations
from typing import List, Optional, Sequence, Set, Union

from yaad import ops
from yaad import node
from yaad import grad_mode


def topological_order(root: Optional[ops.Operator],
                      required_ops: Optional[Set[ops.Operator]],
                      _order=None,
                      _visited=None) -> List[ops.Operator]:
    if root is None or (required_ops is not None and root not in required_ops):
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


def prune_graph(inputs: Sequence[node.Node], outputs: Sequence[node.Node]):
    def select_nodes(root: ops.Operator,
                     inputs: Sequence[node.Node],
                     _visited: Optional[Set] = None,
                     _required: Optional[Set] = None) -> bool:
        _visited = set() if _visited is None else _visited
        _required = set() if _required is None else _required
        if root is None:
            return _required
        _visited.add(root)
        if isinstance(root, ops.LeafOp) and root.variable in inputs:
            _required.add(root)
            return _required
        for child in root.next_ops:
            _required = select_nodes(child, inputs, _visited, _required)
            if child in _required:
                _required.add(root)
        return _required
    required = set()
    visited = set()
    for output in outputs:
        required = select_nodes(output.op, inputs, visited, required)

# TODO: What to do when there is no path from output to input? zero or none?
# torch throws an error.
# When you know the outputs and the inputs, you don't need to traverse the full
# graph? find all paths from out to in and mark nodes. then only traverse
# those nodes.


def backward(output: node.Node,
             grad_output=None,
             retain_graph=False):
    with grad_mode.set_grad_enabled(False):
        grad_output = (node.Node(1.)
                       if grad_output is None else grad_output)
        output.op.grad_store.update(grad_output)
        ordered_ops = topological_order(output.op)
        autodiff(ordered_ops, retain_graph, accumulate_grad=True)

# TODO: finish implementing the grad function.


def grad(outputs: Union[node.Node, Sequence[node.Node]],
         inputs: Union[node.Node, Sequence[node.Node]],
         grad_outputs: Union[node.Node, Sequence[node.Node], None] = None,
         retain_graph: bool = False,
         create_graph: bool = False):
    inputs = [inputs] if isinstance(inputs, node.Node) else inputs
    outputs = [outputs] if isinstance(outputs, node.Node) else outputs
    grad_outputs = (
        [grad_outputs] if isinstance(grad_outputs, node.Node) else grad_outputs)
    with grad_mode.set_grad_enabled(create_graph):
        # TODO: loop over all grad outs.
        grad_output = (node.Node(1., requires_grad=create_graph)
                       if grad_output is None else grad_output)
        required_ops = prune_graph(inputs, outputs)
        # TODO: loop over all outputs
        outputs.op.grad_store.update(grad_output)
        # TODO: loop over all outputs
        ordered_ops = topological_order(outputs.op, required_ops)
        # TODO: loop over all outputs ?
        # what happens when 2 outputs share a subgraph?
        # set retain graph to true for all but last call?
        # autodiff should accept inputs and return grads.
        # grads same size as inputs but none/0 for when not computed.
        # remember fm-bm discussion will need to loop through as many times
        # as the number of outputs. in forward more this will be more
        # efficient.
        autodiff(ordered_ops, retain_graph, accumulate_grad=False)


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
