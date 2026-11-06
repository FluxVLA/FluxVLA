"""Structural contract tests; stand-ins are not numerical model evidence."""

from __future__ import annotations
import unittest

from fluxvla_export import (ContractError, PreparedDeployment, PreparedGraph,
                            TensorSpec)


class FakeTensor:

    def __init__(self, shape, dtype='float32'):
        self.shape = shape
        self.dtype = dtype

    def detach(self):
        return self


class FakeModule:

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def forward(self, *inputs):
        return inputs[0]

    def eval(self):
        return self

    def state_dict(self):
        return {}


class DummyEngine:

    def infer(self, inputs, *, synchronize=False):
        return {'actions': inputs['images']}


class DummyExecutor(DummyEngine):
    pass


class ContractTest(unittest.TestCase):

    @staticmethod
    def graph(name='model'):
        return PreparedGraph(
            name=name,
            module=FakeModule(),
            inputs={'images': FakeTensor((1, 3, 224, 224))},
            input_specs=(TensorSpec('images', 'float32', (1, 3, 224, 224)), ),
            output_specs=(TensorSpec('features', 'float32', (1, 32)), ),
        )

    @staticmethod
    def deployment(**overrides):
        graph = ContractTest.graph()
        values = {
            'recipe':
            'test-v1',
            'graphs': {
                'model': graph,
            },
            'policy_inputs': {
                'images': FakeTensor((1, 3, 224, 224)),
            },
            'policy_input_specs': (TensorSpec('images', 'float32',
                                              (1, 3, 224, 224)), ),
            'output_specs': (TensorSpec('actions', 'float32', (1, 10, 7)), ),
            'reference_runner':
            lambda: {
                'actions': FakeTensor((1, 10, 7)),
            },
            'executor_factory':
            lambda engines: DummyExecutor(),
        }
        values.update(overrides)
        return PreparedDeployment(**values)

    def test_rejects_non_integer_dimensions(self):
        for dimension in (1.9, '1', True):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(ContractError, 'integers'):
                    TensorSpec('bad', 'float32', (dimension, 3))

    def test_dynamic_shape_requires_complete_profile(self):
        with self.assertRaisesRegex(ContractError, 'min/opt/max'):
            TensorSpec(
                'tokens',
                'int32',
                (1, -1),
                min_shape=(1, 8),
            )

    def test_static_dimension_must_match_profile(self):
        with self.assertRaisesRegex(ContractError, 'static dimension'):
            TensorSpec(
                'tokens',
                'int32',
                (1, -1, 32),
                min_shape=(1, 8, 16),
                opt_shape=(1, 16, 16),
                max_shape=(1, 32, 16),
            )

    def test_valid_dynamic_profile_accepts_inclusive_bounds(self):
        spec = TensorSpec(
            'x',
            'float32', (-1, 3),
            min_shape=(1, 3),
            opt_shape=(2, 3),
            max_shape=(4, 3))
        for batch in (1, 2, 4):
            with self.subTest(batch=batch):
                graph = PreparedGraph(
                    name='profile',
                    module=FakeModule(),
                    inputs={'x': FakeTensor((batch, 3))},
                    input_specs=(spec, ),
                    output_specs=(spec, ))
                self.assertEqual(graph.input_specs[0], spec)

    def test_dynamic_input_outside_profile_is_rejected(self):
        spec = TensorSpec(
            'x',
            'float32', (-1, 3),
            min_shape=(2, 3),
            opt_shape=(3, 3),
            max_shape=(4, 3))
        for batch in (1, 5):
            with self.subTest(batch=batch):
                with self.assertRaisesRegex(ContractError, 'outside'):
                    PreparedGraph(
                        name='profile',
                        module=FakeModule(),
                        inputs={'x': FakeTensor((batch, 3))},
                        input_specs=(spec, ),
                        output_specs=(spec, ))

    def test_reference_accepts_matching_output_contract(self):
        outputs = self.deployment().run_reference()
        self.assertEqual(tuple(outputs), ('actions', ))
        self.assertEqual(outputs['actions'].shape, (1, 10, 7))
        self.assertEqual(outputs['actions'].dtype, 'float32')

    def test_reference_rejects_wrong_output_names(self):
        deployment = self.deployment(
            reference_runner=lambda: {'wrong_name': FakeTensor((1, 10, 7))})
        with self.assertRaisesRegex(ContractError, 'Reference outputs'):
            deployment.run_reference()

    def test_reference_rejects_wrong_output_dtype(self):
        deployment = self.deployment(
            reference_runner=lambda:
            {'actions': FakeTensor((1, 10, 7), 'float16')})
        with self.assertRaisesRegex(ContractError, 'dtype'):
            deployment.run_reference()

    def test_reference_requires_declared_output_order(self):
        specs = (TensorSpec('first', 'float32',
                            (1, 3)), TensorSpec('second', 'float32', (1, 3)))
        deployment = self.deployment(
            output_specs=specs,
            reference_runner=lambda: {
                'second': FakeTensor((1, 3)),
                'first': FakeTensor((1, 3)),
            })
        with self.assertRaisesRegex(ContractError, 'Reference outputs'):
            deployment.run_reference()

    def test_graph_requires_module_surface(self):
        with self.assertRaisesRegex(ContractError, 'state_dict'):
            PreparedGraph(
                name='model',
                module=lambda images: images,
                inputs={'images': FakeTensor((1, 3))},
                input_specs=(TensorSpec('images', 'float32', (1, 3)), ),
                output_specs=(TensorSpec('features', 'float32', (1, 3)), ),
            )

    def test_graph_requires_input_order_to_match_specs(self):
        with self.assertRaisesRegex(ContractError, 'input order'):
            PreparedGraph(
                name='model',
                module=FakeModule(),
                inputs={
                    'images': FakeTensor((1, 3, 224, 224)),
                    'tokens': FakeTensor((1, 16), 'int32'),
                },
                input_specs=(
                    TensorSpec('tokens', 'int32', (1, 16)),
                    TensorSpec('images', 'float32', (1, 3, 224, 224)),
                ),
                output_specs=(TensorSpec('features', 'float32', (1, 32)), ),
            )

    def test_graph_validates_input_dtype(self):
        with self.assertRaisesRegex(ContractError, 'dtype'):
            PreparedGraph(
                name='model',
                module=FakeModule(),
                inputs={'images': FakeTensor((1, 3), 'float16')},
                input_specs=(TensorSpec('images', 'float32', (1, 3)), ),
                output_specs=(TensorSpec('features', 'float32', (1, 3)), ),
            )

    def test_deployment_validates_reference_shape(self):
        deployment = self.deployment(reference_runner=lambda: {
            'actions': FakeTensor((1, 9, 7)),
        })
        with self.assertRaisesRegex(ContractError, 'dimension 1'):
            deployment.run_reference()

    def test_deployment_requires_exact_engine_names(self):
        deployment = self.deployment()
        with self.assertRaisesRegex(ContractError, 'Engine names'):
            deployment.create_executor({'unexpected': DummyEngine()})

    def test_deployment_rejects_non_callable_engine_infer(self):

        class InvalidEngine:
            infer = 1

        deployment = self.deployment()
        with self.assertRaisesRegex(ContractError, 'callable infer'):
            deployment.create_executor({'model': InvalidEngine()})

    def test_deployment_rejects_non_callable_executor_infer(self):

        class InvalidExecutor:
            infer = 1

        deployment = self.deployment(
            executor_factory=lambda engines: InvalidExecutor())
        with self.assertRaisesRegex(ContractError, 'callable infer'):
            deployment.create_executor({'model': DummyEngine()})

    def test_deployment_normalizes_engine_order(self):
        received = []

        def create_executor(engines):
            received.extend(engines)
            return DummyExecutor()

        deployment = PreparedDeployment(
            recipe='test-v1',
            graphs={
                'first': self.graph('first'),
                'second': self.graph('second'),
            },
            policy_inputs={'images': FakeTensor((1, 3, 224, 224))},
            policy_input_specs=(TensorSpec('images', 'float32',
                                           (1, 3, 224, 224)), ),
            output_specs=(TensorSpec('actions', 'float32', (1, 10, 7)), ),
            reference_runner=lambda: {
                'actions': FakeTensor((1, 10, 7)),
            },
            executor_factory=create_executor,
        )
        deployment.create_executor({
            'second': DummyEngine(),
            'first': DummyEngine(),
        })
        self.assertEqual(received, ['first', 'second'])


if __name__ == '__main__':
    unittest.main()
