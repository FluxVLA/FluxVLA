"""Dependency-light policy declarations shared by driver and Ray workers.

Add supported models here, not to train/eval/worker dispatch branches.
Import paths keep unrelated model implementations out of builder imports.
"""

from dataclasses import dataclass

from fluxvla.rl.utils.imports import resolve_symbol


@dataclass(frozen=True)
class PolicySpec:
    model_type: str
    native_model_type: str
    policy_class: str
    horizon_config_key: str
    observation_adapters: tuple
    builder: str = 'fluxvla.rl.models.builder.build_flow_policy'
    checkpoint_resolver: str = None
    fsdp_layers: tuple = None
    fsdp_modules: tuple = None
    fsdp_note: str = ''
    match_rollout_microbatch: bool = False

    def validate_adapter(self, adapter_name):
        if adapter_name not in self.observation_adapters:
            supported = ', '.join(self.observation_adapters)
            raise ValueError(
                f'{self.model_type} supports adapters: {supported}. '
                f'Got {adapter_name}')

    def validate_frontend(self, cfg):
        adapter_name = cfg.actor.model.fluxvla.get('observation_adapter',
                                                   'libero')
        self.validate_adapter(adapter_name)
        for phase in ('train', 'eval'):
            if cfg.env[phase].env_type != adapter_name:
                raise ValueError(
                    f'{phase} environment must match adapter {adapter_name}')
        wrap = cfg.actor.fsdp_config.wrap_policy
        for key, expected in (
            ('transformer_layer_cls_to_wrap', self.fsdp_layers),
            ('module_classes_to_wrap', self.fsdp_modules),
        ):
            if expected is not None and tuple(wrap.get(key, ())) != expected:
                raise ValueError(
                    f'Use the {self.model_type} FSDP wrap policy. '
                    f'{self.fsdp_note}')
        if self.match_rollout_microbatch and (
                cfg.actor.model.fluxvla.get('rollout_micro_batch_size') !=
                cfg.actor.micro_batch_size):
            raise ValueError(
                f'{self.model_type} rollout and actor microbatch must match')


POLICY_SPECS = {
    spec.model_type: spec
    for spec in (
        PolicySpec(
            model_type='fluxvla_pi05',
            native_model_type='PI05FlowMatching',
            policy_class='fluxvla.rl.models.pi05.policy.FluxPI05RLPolicy',
            horizon_config_key='n_action_steps',
            observation_adapters=('libero', 'robotwin'),
            checkpoint_resolver=(
                'fluxvla.rl.models.pi05.checkpoint.resolve_candidates'),
            fsdp_layers=('GemmaDecoderLayer', 'SiglipEncoderLayer'),
            fsdp_modules=('LinearProjector', 'ValueHead')),
        PolicySpec(
            model_type='fluxvla_smolvla',
            native_model_type='SmolVLAFlowMatching',
            policy_class=(
                'fluxvla.rl.models.smolvla.policy.FluxSmolVLARLPolicy'),
            horizon_config_key='chunk_size',
            observation_adapters=('libero', ),
            fsdp_layers=('SmolVLMEncoderLayer', ),
            fsdp_modules=('LinearProjector', 'ValueHead'),
            fsdp_note='interleaved decoder forwards bypass hooks',
            match_rollout_microbatch=True),
    )
}


def get_policy_spec(model_type):
    try:
        return POLICY_SPECS[model_type]
    except KeyError as error:
        raise ValueError(
            f'Unknown FluxVLA RL model type: {model_type}') from error


def checkpoint_resolver_for(model):
    """Support loading native SFT models as well as their RL subclasses."""
    for cls in type(model).__mro__:
        for spec in POLICY_SPECS.values():
            if cls.__name__ == spec.native_model_type:
                return (resolve_symbol(spec.checkpoint_resolver)
                        if spec.checkpoint_resolver else None)
    return None
