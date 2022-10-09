from clove import binding_utils as _binding_utils
from clove import ops as _ops

clone = _binding_utils.make_fn(_ops.CloneOp)
transpose = _binding_utils.make_fn(_ops.TransposeOp)
permute = _binding_utils.make_fn(_ops.PermuteOp)
add = _binding_utils.make_fn(_ops.AddOp)
multiply = _binding_utils.make_fn(_ops.MulOp)
matmul = _binding_utils.make_fn(_ops.MatmulOp)
negative = _binding_utils.make_fn(_ops.NegOp)
subtract = _binding_utils.make_fn(_ops.MinusOp)
exp = _binding_utils.make_fn(_ops.ExpOp)
log = _binding_utils.make_fn(_ops.LogOp)
power = _binding_utils.make_fn(_ops.PowOp)
sigmoid = _binding_utils.make_fn(_ops.SigmoidOp)
tanh = _binding_utils.make_fn(_ops.TanhOp)
