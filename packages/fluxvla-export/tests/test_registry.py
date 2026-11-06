from __future__ import annotations

import unittest
import uuid

from fluxvla_export import (ContractError, DuplicateRecipeError,
                            RecipeNotFoundError)
from fluxvla_export.registry import get_recipe, register_recipe


class RegistryTest(unittest.TestCase):

    def test_registers_and_constructs_recipe(self):
        recipe_name = f'test-{uuid.uuid4().hex}'

        class Recipe:
            name = recipe_name

            def prepare(self, *, config, checkpoint, inputs, device):
                raise AssertionError('not called')

        register_recipe(recipe_name)(Recipe)
        self.assertIsInstance(get_recipe(recipe_name), Recipe)

    def test_rejects_duplicate_name(self):
        recipe_name = f'test-{uuid.uuid4().hex}'

        class Recipe:
            name = recipe_name

            def prepare(self, *, config, checkpoint, inputs, device):
                raise AssertionError('not called')

        register_recipe(recipe_name)(Recipe)
        with self.assertRaisesRegex(
                DuplicateRecipeError, 'already registered'):
            register_recipe(recipe_name)(Recipe)

    def test_rejects_non_callable_prepare(self):
        recipe_name = f'test-{uuid.uuid4().hex}'

        class InvalidRecipe:
            name = recipe_name
            prepare = 1

        register_recipe(recipe_name)(InvalidRecipe)
        with self.assertRaisesRegex(ContractError, 'callable prepare'):
            get_recipe(recipe_name)

    def test_unknown_recipe_lists_error(self):
        recipe_name = f'missing-{uuid.uuid4().hex}'
        with self.assertRaisesRegex(RecipeNotFoundError, recipe_name):
            get_recipe(recipe_name)


if __name__ == '__main__':
    unittest.main()
