"""PI0.5 MCore TP2 recipe.

Launch with two ranks.  Linear projections are output-partitioned and the
remaining replicated parameters synchronize gradients over the TP group.
"""

_base_ = ['./pi05_paligemma_libero_10_mcore.py']

runner = dict(
    tensor_model_parallel_size=2,
    tensor_parallel_min_out_features=128,
)
