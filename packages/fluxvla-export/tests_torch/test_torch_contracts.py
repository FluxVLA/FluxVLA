"""Real CPU tensor/module handoff, not a model recipe or TensorRT test."""

import unittest

import torch

from fluxvla_export import (ContractError, PreparedDeployment, PreparedGraph,
                            TensorSpec)


class TorchEngine:
    """Test-only eager PyTorch implementation of the engine protocol."""

    def __init__(self, module):
        self.module = module

    def infer(self, inputs, *, synchronize=False):
        with torch.no_grad():
            return {'y': self.module(inputs['x'])}


class OneGraphExecutor:

    def __init__(self, engines):
        self.engine = engines['linear']

    def infer(self, inputs, *, synchronize=False):
        return self.engine.infer(inputs, synchronize=synchronize)


class TorchContractTest(unittest.TestCase):

    def setUp(self):
        self.module = torch.nn.Linear(3, 2).eval()
        with torch.no_grad():
            self.module.weight.copy_(
                torch.tensor([[1., 0., -1.], [.5, 2., 0.]]))
            self.module.bias.copy_(torch.tensor([.25, -.5]))
        self.x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        self.expected = torch.tensor([[-1.75, 4.], [-1.75, 11.5]])
        self.input_spec = TensorSpec('x', 'float32', (2, 3))
        self.output_spec = TensorSpec('y', 'float32', (2, 2))

    def graph(self, x=None):
        return PreparedGraph(
            name='linear',
            module=self.module,
            inputs={'x': self.x if x is None else x},
            input_specs=(self.input_spec, ),
            output_specs=(self.output_spec, ))

    def test_real_reference_and_executor_preserve_handoff_identity(self):
        graph = self.graph()
        self.assertIs(graph.module, self.module)
        self.assertIs(graph.inputs['x'], self.x)
        self.assertFalse(graph.module.training)
        original = self.x.clone()
        engine = TorchEngine(self.module)
        deployment = PreparedDeployment(
            recipe='contract-test-only',
            graphs={'linear': graph},
            policy_inputs={'x': self.x},
            policy_input_specs=(self.input_spec, ),
            output_specs=(self.output_spec, ),
            reference_runner=lambda: engine.infer({'x': self.x}),
            executor_factory=OneGraphExecutor)
        reference = deployment.run_reference()['y']
        executor = deployment.create_executor({'linear': engine})
        self.assertIs(executor.engine, engine)
        actual = executor.infer({'x': self.x}, synchronize=True)['y']
        torch.testing.assert_close(reference, self.expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, self.expected, rtol=0, atol=0)
        torch.testing.assert_close(self.x, original, rtol=0, atol=0)
        self.assertFalse(reference.requires_grad)

    def test_real_tensor_contract_mismatches_fail(self):
        with self.assertRaisesRegex(ContractError, 'dtype'):
            self.graph(self.x.to(torch.float64))
        with self.assertRaisesRegex(ContractError, 'dimension'):
            self.graph(self.x[:, :2])

    def test_real_reference_output_shape_is_checked(self):
        deployment = PreparedDeployment(
            recipe='contract-test-only',
            graphs={'linear': self.graph()},
            policy_inputs={'x': self.x},
            policy_input_specs=(self.input_spec, ),
            output_specs=(self.output_spec, ),
            reference_runner=lambda: {'y': self.module(self.x)[:, :1]},
            executor_factory=OneGraphExecutor)
        with self.assertRaisesRegex(ContractError, 'dimension'):
            deployment.run_reference()


if __name__ == '__main__':
    unittest.main()
