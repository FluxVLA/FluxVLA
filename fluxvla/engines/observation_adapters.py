# Copyright 2026 Limx Dynamics
"""Environment-side facts; no model inference or tokenization."""

from .utils.root import OBSERVATION_ADAPTERS


@OBSERVATION_ADAPTERS.register_module()
class RobocasaDrawerObservationAdapter:
    """Attach simulator-grounded drawer predicates to a copied observation."""

    def __call__(self, observation, env):
        from robocasa.utils.object_utils import obj_inside_of

        simulator = env.unwrapped.env
        if getattr(simulator, 'behavior', None) != 'close':
            raise ValueError('Drawer planning requires a drawer-close task')
        signals = simulator.get_subtask_term_signals()
        facts = {
            'grasped:cup':
            bool(signals['grasp_object']),
            'inside:cup:drawer':
            bool(
                obj_inside_of(
                    env=simulator,
                    obj_name=simulator.objects['obj'].name,
                    fixture_id=simulator.drawer,
                    partial_check=True)),
            'closed:drawer':
            bool(
                simulator.drawer.get_door_state(
                    env=simulator)['door'] <= 0.005),
        }
        return dict(observation, predicates=facts)
