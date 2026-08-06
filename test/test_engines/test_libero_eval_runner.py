# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from unittest.mock import patch

from fluxvla.datasets.transformed_eval_dataset import TransformedEvalDataset
from fluxvla.engines.runners.libero_eval_runner import LiberoEvalRunner
from fluxvla.transforms.normalize import PostprocessLiberoAction


class TestLiberoEvalRunnerContracts(unittest.TestCase):

    def test_component_context_defaults_preserve_legacy_contract(self):
        class LegacyComponent:
            pass

        defaults = ('task_suite_name', 'norm_stats_key', 'norm_stats')
        self.assertEqual(
            LiberoEvalRunner._get_eval_context_keys(
                LegacyComponent, defaults), defaults)
        self.assertEqual(
            LiberoEvalRunner._get_eval_context_keys(
                TransformedEvalDataset, defaults), ())
        self.assertEqual(
            LiberoEvalRunner._get_eval_context_keys(
                PostprocessLiberoAction, ('norm_stats',)), ())

    @patch('fluxvla.engines.runners.libero_eval_runner.dist.is_initialized')
    @patch('fluxvla.engines.runners.libero_eval_runner.dist.is_available')
    def test_collectives_require_initialized_multi_process_group(
            self, is_available, is_initialized):
        is_available.return_value = True
        is_initialized.return_value = False
        self.assertFalse(
            LiberoEvalRunner._distributed_collectives_enabled(1))
        self.assertFalse(
            LiberoEvalRunner._distributed_collectives_enabled(2))

        is_initialized.return_value = True
        self.assertTrue(
            LiberoEvalRunner._distributed_collectives_enabled(2))


if __name__ == '__main__':
    unittest.main()