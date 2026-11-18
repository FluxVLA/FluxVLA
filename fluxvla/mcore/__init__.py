"""Native Megatron-Core backend contracts and model recipes.

Import a concrete recipe, such as ``fluxvla.mcore.pi05.recipe.PI05_SPEC``,
instead of importing a model through this package root.
"""

from .contracts import MCoreModelSpec

__all__ = ["MCoreModelSpec"]
