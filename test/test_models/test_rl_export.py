"""Exercise the RL export -> normal FluxVLA pretrained-loading path."""

import importlib.util
from pathlib import Path
from test.test_models.test_base_vla_checkpoint import _TinyVLA, _weights

import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    'rl_export',
    Path(__file__).resolve().parents[2] / 'scripts/rl_export.py')
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
export_policy = _module.export_policy


@pytest.mark.parametrize('wrapper', [None, 'model', 'state_dict'])
@pytest.mark.parametrize('use_directory', [True, False])
def test_rl_export_loads_with_explicit_native_mapping(tmp_path, wrapper,
                                                      use_directory):
    step = tmp_path / 'global_step_10'
    checkpoint = step / 'actor/model_state_dict/full_weights.pt'
    checkpoint.parent.mkdir(parents=True)
    state = _weights()
    state['value_head.weight'] = torch.ones(2, 3)
    state['value_head.bias'] = torch.ones(2)
    torch.save({wrapper: state} if wrapper else state, checkpoint)
    output = tmp_path / 'policy.pt'
    assert export_policy(step if use_directory else checkpoint,
                         output) == (3, 2)
    exported = torch.load(output, weights_only=True)
    assert set(exported) == set(_weights())
    for name, value in exported.items():
        assert value.dtype == state[name].dtype
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    model = _TinyVLA(pretrained_name_or_path=str(output), name_mapping=None)
    model.from_pretrained()
    torch.testing.assert_close(
        model(torch.ones(1, 3)), torch.tensor([[4.0, 26.0]]), rtol=0, atol=0)
    # Export must not modify the training checkpoint or overwrite an export.
    original = torch.load(checkpoint, weights_only=True)
    assert 'value_head.weight' in (original[wrapper] if wrapper else original)
    with pytest.raises(FileExistsError):
        export_policy(checkpoint, output)


@pytest.mark.parametrize('state', [
    {},
    {
        'value_head.weight': torch.ones(1)
    },
    {
        'optimizer': {}
    },
    {
        '_flat_param': torch.ones(1)
    },
])
def test_export_rejects_invalid_state(tmp_path, state):
    path = tmp_path / 'bad.pt'
    torch.save(state, path)
    output = tmp_path / 'policy.pt'
    with pytest.raises(ValueError):
        export_policy(path, output)
    assert not output.exists()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('openpi_rl', [False, True])
def test_pi05_export_preserves_eval_actions(tmp_path, dtype, openpi_rl):
    pytest.importorskip('rlinf')
    from mmengine import Config

    from fluxvla.engines import build_vla_from_cfg
    from fluxvla.models.vlas.pi05_rl_builder import \
        _preserve_legacy_rl_numerics
    from fluxvla.models.vlas.pi05_rl_policy import FluxPI05RLPolicy

    root = Path(__file__).resolve().parents[2]
    cfg = Config.fromfile(
        str(root /
            'configs/pi05/pi05_paligemma_libero_10_full_finetune.py')).model
    # Retain the production name_mapping while shrinking real model modules.
    cfg.pretrained_name_or_path = None
    cfg.strict_mapping = True
    cfg.proj_width = 16
    cfg.n_action_steps = cfg.num_steps = 3
    # Ordinary inference uses this flag to cast Gemma inputs, whereas the RL
    # policy uses compute_dtype. Match both as well as the autocast context.
    cfg.enable_mixed_precision_training = dtype == torch.bfloat16
    cfg.preserve_fp32_residuals = False
    # Ordinary evaluation must explicitly copy both switches from the RL run.
    cfg.openpi_fp32_flow = openpi_rl
    cfg.openpi_rl_numerics = openpi_rl
    cfg.vision_backbone.openpi_stem_fp32 = openpi_rl
    cfg.vision_backbone.openpi_rl_numerics = openpi_rl
    for key, hidden in (('llm_backbone', 32), ('llm_expert', 16)):
        block = cfg[key]
        block.hidden_size = hidden
        block.intermediate_size = hidden * 2
        block.head_dim = 8
        block.num_attention_heads = 2
        block.num_key_value_heads = 1
        block.num_hidden_layers = 2
        block.vocab_size = 64
        block.adarms_cond_dim = None if key == 'llm_backbone' else 16
    vision = cfg.vision_backbone.vision_config
    vision.hidden_size = 16
    vision.intermediate_size = 32
    vision.num_attention_heads = 2
    vision.num_hidden_layers = 1
    vision.image_size = 16
    vision.patch_size = 8
    vision.projection_dim = 32
    cfg.projector.in_dim, cfg.projector.out_dim = 16, 32
    cfg.action_in_proj.out_dim = 16
    cfg.action_out_proj.in_dim = 16
    for key in ('time_mlp_in', 'time_mlp_out'):
        cfg[key].in_dim = cfg[key].out_dim = 16

    class PreparedObservations:

        def __call__(self, obs):
            return obs

        def env_actions(self, actions, chunk, dim, **kwargs):
            return actions[:, :chunk, :dim]

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        policy_cfg = cfg.deepcopy()
        # Emulate a saved RL config from before these opt-in flags existed.
        policy_cfg.pop('openpi_rl_numerics')
        policy_cfg.vision_backbone.pop('openpi_rl_numerics')
        _preserve_legacy_rl_numerics(policy_cfg)
        policy_cfg.type = FluxPI05RLPolicy
        policy = build_vla_from_cfg(policy_cfg).float()
        policy.configure_rl(
            PreparedObservations(), action_chunk=2, compute_dtype=dtype)
        checkpoint = tmp_path / 'full_weights.pt'
        torch.save(policy.state_dict(), checkpoint)
        exported = tmp_path / 'policy.pt'
        export_policy(checkpoint, exported)
        # Exported tensors use Flux names: explicitly select native loading.
        native_cfg = cfg.deepcopy()
        native_cfg.name_mapping = None
        model = build_vla_from_cfg(native_cfg).float().eval()
        model.pretrained_name_or_path = str(exported)
        model.from_pretrained()
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, policy.state_dict()[name], rtol=0, atol=0)
        obs = {
            'images': torch.randn(2, 6, 16, 16),
            'img_masks': torch.ones(2, 2, dtype=torch.bool),
            'lang_tokens': torch.randint(0, 64, (2, 4)),
            'lang_masks': torch.ones(2, 4, dtype=torch.bool),
            'states': torch.randn(2, 32),
        }
        noise = torch.randn(2, 3, 32)
        _, result = policy.predict_action(obs, mode='eval', noise=noise)
        with torch.no_grad(), policy._network_context():
            actual = model.predict_action(**obs, noise=noise.clone())
        torch.testing.assert_close(
            actual,
            result['forward_inputs']['model_action'].reshape_as(actual),
            rtol=0,
            atol=0)
    finally:
        torch.set_num_threads(threads)
