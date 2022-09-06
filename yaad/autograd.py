
from __future__ import annotations
from typing import Dict, List, Optional, Sequence, Set, Union

from yaad import ops
from yaad import node
from yaad import grad_mode


def topological_order(root: Optional[ops.Operator],
                      required_ops: Optional[Set[ops.Operator]],
                      _order=None,
                      _visited=None) -> List[ops.Operator]:
    _visited = set() if _visited is None else _visited
    _order = [] if _order is None else _order
    if root is None or (required_ops is not None and root not in required_ops):
        return _order
    if root in _visited:
        return _order
    _visited.add(root)
    for op in root._children:
        _order = topological_order(op, required_ops, _order, _visited)
    _order.append(root)
    return _order


def prune_graph(inputs: Sequence[node.Node], outputs: Sequence[node.Node]):
    def select_nodes(root: ops.Operator,
                     inputs: Sequence[node.Node],
                     _visited: Optional[Set] = None,
                     _selected: Optional[Set] = None) -> bool:
        _visited = set() if _visited is None else _visited
        _selected = set() if _selected is None else _selected
        if root is None:
            return _selected
        _visited.add(root)
        if isinstance(root, ops.LeafOp) and root.variable in inputs:
            _selected.add(root)
            return _selected
        for child in root.next_ops:
            _selected = select_nodes(child, inputs, _visited, _selected)
            if child in _selected:
                _selected.add(root)
        return _selected
    selected = set()
    visited = set()
    for output in outputs:
        selected = select_nodes(output.op, inputs, visited, selected)
    return selected


def backward(output: node.Node,
             grad_output=None,
             retain_graph=False):
    with grad_mode.set_grad_enabled(False):
        grad_output = (node.Node(1.)
                       if grad_output is None else grad_output)
        output.op.grad_store.update(grad_output)
        ordered_ops = topological_order(output.op)
        autodiff(ordered_ops, retain_graph, populate_grad=True)


def grad(outputs: Union[node.Node, Sequence[node.Node]],
         inputs: Union[node.Node, Sequence[node.Node]],
         grad_outputs: Union[node.Node, Sequence[node.Node], None] = None,
         retain_graph: bool = False,
         create_graph: bool = False):
    inputs = [inputs] if isinstance(inputs, node.Node) else inputs
    outputs = [outputs] if isinstance(outputs, node.Node) else outputs
    grad_outputs = ([grad_outputs]
                    if isinstance(grad_outputs, node.Node)
                    else [None] * len(outputs) if grad_outputs is None
                    else grad_outputs)
    if not len(grad_outputs) == len(outputs):
        raise ValueError(
            "Expected grad outputs to be the same length as outputs. Found"
            f" lengths {len(grad_outputs)} and {len(outputs)} instead. ")
    with grad_mode.set_grad_enabled(create_graph):
        for i, (out, g_out) in enumerate(zip(outputs, grad_outputs)):
            # TODO: changed requires grad here to fix memory leaks.
            # seems to have triggered another side effect. investigate.
            grad_outputs[i] = (node.Node(1.)
                               if g_out is None else g_out)
            out.op.grad_store.update(grad_outputs[i])
        required_ops = prune_graph(inputs, outputs)
        grad_map = None
        for out in outputs:
            # TODO: these multiple calls to topo_order can be optimized.
            # by sending in visited and order. but need to weed out
            # ops that are not part of this graph.
            ordered_ops = topological_order(out.op, required_ops)
            grad_map = autodiff(
                ordered_ops,
                retain_graph=retain_graph if out is outputs[-1] else True,
                populate_grad=False,
                inputs=inputs,
                grad_map=grad_map)
    return tuple(grad_map.values())


def autodiff(ordered_ops: List[ops.Operator],
             retain_graph: bool,
             populate_grad=True,
             inputs=None,
             grad_map: Dict[node.Node, node. Node] = None):
    if not populate_grad and inputs is None:
        raise ValueError(
            "Must provide a sequence of inputs to differentiate with respect"
            " to when populate_grad is set to `False`.")
    if inputs is not None and grad_map is None:
        grad_map = {inp: None for inp in inputs}
    op = ordered_ops.pop()
    grad_output = op.grad_store.value
    op.grad_store.reset()
    output = op.variable
    if (populate_grad
            and output is not None
            and (output.is_leaf() or output.retains_grad)):
        output.grad = (grad_output if output.grad is None
                       else output.grad + grad_output)
    elif output is not None and output.is_leaf() and output in grad_map:
        grad_map[output] = (grad_output if grad_map[output] is None
                            else grad_map[output] + grad_output)
    grads = op.backward(grad_output)
    grads = (grads,) if isinstance(grads, node.Node) else grads
    for child, grad in zip(op.next_ops, grads):
        if child is not None:
            child.grad_store.update(grad)
    if not retain_graph:
        op.clear_cache()
    if ordered_ops:
        autodiff(ordered_ops, retain_graph, populate_grad, inputs, grad_map)
    return grad_map
