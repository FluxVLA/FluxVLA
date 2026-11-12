"""PI0.5 MCore data-parallel recipe with sharded optimizer state."""

_base_ = ['./pi05_paligemma_libero_10_mcore.py']

runner = dict(distributed_optimizer=True)
