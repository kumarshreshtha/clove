
from clove import binding_utils
from clove import ops

clone = binding_utils.make_fn('clone', ops.CloneOp)
transpose = binding_utils.make_fn('transpose', ops.TransposeOp)
permute = binding_utils.make_fn('permute', ops.PermuteOp)
add = binding_utils.make_fn('add', ops.AddOp)
multiply = binding_utils.make_fn('multiply', ops.MulOp)
matmul = binding_utils.make_fn('matmul', ops.MatmulOp)
negative = binding_utils.make_fn('negative', ops.NegOp)
subtract = binding_utils.make_fn('subtract', ops.MinusOp)
exp = binding_utils.make_fn('exp', ops.ExpOp)
log = binding_utils.make_fn('log', ops.LogOp)
power = binding_utils.make_fn('power', ops.PowOp)
sigmoid = binding_utils.make_fn('sigmoid', ops.SigmoidOp)
tanh = binding_utils.make_fn('tanh', ops.TanhOp)