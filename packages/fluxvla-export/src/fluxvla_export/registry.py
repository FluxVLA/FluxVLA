# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import Callable

from .contracts import ExportRecipe
from .errors import (ContractError, DuplicateRecipeError,
                     RecipeNotFoundError)

RecipeFactory = Callable[[], ExportRecipe]
_RECIPE_FACTORIES: dict[str, RecipeFactory] = {}


def register_recipe(name: str):
    """Register one built-in recipe factory under a stable public name."""
    if not name or name.strip() != name:
        raise ContractError(
            'Recipe names must be non-empty and cannot have surrounding '
            'spaces.')

    def decorator(factory: RecipeFactory) -> RecipeFactory:
        if name in _RECIPE_FACTORIES:
            raise DuplicateRecipeError(
                f'Export recipe {name!r} is already registered.')
        _RECIPE_FACTORIES[name] = factory
        return factory

    return decorator


def available_recipes() -> tuple[str, ...]:
    """Return registered recipe names in deterministic order."""
    return tuple(sorted(_RECIPE_FACTORIES))


def get_recipe(name: str) -> ExportRecipe:
    """Construct one registered recipe or fail with available names."""
    try:
        factory = _RECIPE_FACTORIES[name]
    except KeyError as error:
        available = ', '.join(available_recipes()) or '<none>'
        message = (
            f'Unknown export recipe {name!r}; available: {available}.')
        raise RecipeNotFoundError(message) from error
    if not callable(factory):
        raise ContractError(f'Recipe factory {name!r} must be callable.')
    recipe = factory()
    if not callable(getattr(recipe, 'prepare', None)):
        raise ContractError(
            f'Recipe factory {name!r} must return an object with callable '
            'prepare().')
    if getattr(recipe, 'name', None) != name:
        raise ContractError(
            f'Recipe factory {name!r} returned recipe name {recipe.name!r}.')
    return recipe
