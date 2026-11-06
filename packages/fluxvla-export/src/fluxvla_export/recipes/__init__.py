"""Built-in model-family registrations; model imports remain lazy."""

from ..registry import register_recipe


@register_recipe('pi05-v1')
def _pi05_recipe():
    from .pi05.recipe import PI05Recipe

    return PI05Recipe()
