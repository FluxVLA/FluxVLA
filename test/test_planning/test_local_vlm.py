# Copyright 2026 Limx Dynamics
"""Local planner contracts; no external API, checkpoint or simulator needed."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from mmengine import Config
from safetensors.torch import save_file

from fluxvla.engines.utils.builder import build_from_cfg
from fluxvla.engines.utils.root import PLANNERS
from fluxvla.planning.local_vlm import (LocalVLMSubtaskPlanner, _CandidateTrie,
                                        resolve_planner_checkpoint)
from fluxvla.transforms import vlm_transforms
from fluxvla.transforms.subtask_vlm_transforms import \
    build_subtask_vlm_messages

CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist']
LABELS = [
    'The left arm grasped the block.',
    'The right arm placed the block on the bowl.'
]
CANDIDATES = [
    dict(choice=f'skill_{i:04d}', instruction=text)
    for i, text in enumerate(LABELS)
]
ROOT = Path(__file__).resolve().parents[2]


class Tokenizer:
    eos_token_id = 1
    all_special_ids = [0, 1]

    def encode(self, text, **kwargs):
        return [ord(char) + 2 for char in text]

    def decode(self, ids, **kwargs):
        return ''.join(chr(token - 2) for token in ids)


class Processor:
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return dict(
            input_ids=torch.tensor([[101, 102]]),
            attention_mask=torch.ones(1, 2, dtype=torch.long),
            pixel_values=torch.zeros(3, 12),
            image_grid_thw=torch.tensor([[1, 2, 2]] * 3),
            mm_token_type_ids=torch.zeros(1, 2, dtype=torch.long))


class Model:

    def __init__(self):
        self.calls = []
        self.text = LABELS[1]
        self.eos = True

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        ids = kwargs['input_ids'][0].tolist()
        target = Tokenizer().encode(self.text) + ([1] if self.eos else [])
        for token in target:
            allowed = kwargs.get('prefix_allowed_tokens_fn')
            if allowed is not None:
                assert token in allowed(0, torch.tensor(ids))
            ids.append(token)
        return torch.tensor([ids])


@pytest.fixture
def local(tmp_path, monkeypatch):
    checkpoint = tmp_path / 'checkpoints/step-1-epoch-1.safetensors'
    checkpoint.parent.mkdir()
    checkpoint.touch()
    assets = tmp_path / 'vlm_assets'
    assets.mkdir()
    (assets / 'config.json').write_text('{}')
    processor, model = Processor(), Model()
    loads = Mock(return_value=processor)
    monkeypatch.setattr(vlm_transforms, 'get_processor_for_vlm', loads)
    model_loads = Mock(return_value=model)
    monkeypatch.setattr(LocalVLMSubtaskPlanner, '_load_model', model_loads)
    planner = build_from_cfg(
        dict(
            type='LocalVLMSubtaskPlanner',
            checkpoint_path=str(checkpoint),
            torch_dtype='fp32',
            device='cpu'), PLANNERS)
    return planner, processor, model, loads, model_loads


def request(candidates=None):
    return dict(
        goal='Stack the blocks.',
        observation={
            name: np.full((4, 4, 3), index, dtype=np.uint8)
            for index, name in enumerate(CAMERAS)
        },
        facts={},
        candidates=candidates or CANDIDATES,
        history=[dict(instruction='do not put history into input')],
        feedback=dict(answer='do not put feedback into input'))


def test_registered_local_backend_matches_training_prompt_and_reuses_weights(
        local):
    planner, processor, model, loads, model_loads = local
    req = request()
    req['observation'].update(
        frame_index=999,
        reward=1,
        success=True,
        annotation='do not expose labels')
    output = planner.choose(**req)
    assert output == dict(choice='skill_0001')
    assert planner.last_prediction['text'] == LABELS[1]
    messages, options = processor.calls[-1]
    assert [message['role'] for message in messages] == ['user']
    content = messages[0]['content']
    assert [part['text'] for part in content if part['type'] == 'text'
            ] == ['Camera: ' + name for name in CAMERAS] + [
                build_subtask_vlm_messages(req['goal'], [
                    part['image']
                    for part in content if part['type'] == 'image'
                ], CAMERAS)[0]['content'][-1]['text']
            ]
    rendered = repr(messages)
    assert all(text not in rendered for text in LABELS)
    assert 'skill_000' not in rendered and 'do not' not in rendered
    assert '999' not in rendered and 'success' not in rendered
    assert options['enable_thinking'] is False
    assert options['add_generation_prompt'] is True
    inputs = model.calls[0]
    assert 'labels' not in inputs
    assert inputs['image_grid_thw'].shape == (3, 3)
    assert inputs['mm_token_type_ids'].shape == (1, 2)
    assert inputs['do_sample'] is False and inputs['num_beams'] == 1
    original_trie = planner._trie
    planner.reset()
    assert planner.last_prediction is None
    planner.choose(**request())
    # Vocabulary cache, not episode state.
    assert planner._trie is original_trie
    assert len(processor.calls) == len(model.calls) == 2
    assert loads.call_count == model_loads.call_count == 1
    assert loads.call_args.kwargs['local_files_only'] is True


def test_task_candidate_switch_does_not_keep_previous_ids(local):
    planner, _, model, _, _ = local
    planner.choose(**request())
    planner.reset()
    changed = [dict(choice='different_task_id', instruction=LABELS[1])]
    assert planner.choose(**request(changed)) == dict(
        choice='different_task_id')
    assert len(model.calls) == 2


@pytest.mark.parametrize('bad_input', ['facts', 'camera', 'goal'])
def test_reject_bad_observation_before_model_call(local, bad_input):
    planner, _, model, _, _ = local
    req = request()
    if bad_input == 'facts':
        req['facts'] = dict(success=True)
    elif bad_input == 'camera':
        req['observation'].pop('cam_left_wrist')
    else:
        req['goal'] = ''
    with pytest.raises(ValueError):
        planner.choose(**req)
    assert not model.calls


def test_reject_generated_text_not_in_candidates_without_fuzzy_fallback(local):
    planner, _, model, _, _ = local
    planner.constrained_decoding = False
    model.text = 'A completely different instruction.'
    with pytest.raises(ValueError, match='no fuzzy fallback'):
        planner.choose(**request())
    assert planner.last_prediction is None


def test_partial_text_without_eos_is_never_executed(local):
    planner, _, model, _, _ = local
    model.eos = False
    with pytest.raises(RuntimeError, match='without EOS'):
        planner.choose(**request())
    assert planner.last_prediction is None


def test_trie_preserves_shared_prefix_and_requires_exact_end():
    tokenizer = Tokenizer()
    candidates = [
        dict(choice='short', instruction='Move'),
        dict(choice='long', instruction='Move left')
    ]
    trie = _CandidateTrie(tokenizer, candidates, 1, 32)
    allowed = trie.allowed_tokens(2)
    prefix = torch.tensor([101, 102] + tokenizer.encode('Move'))
    assert set(allowed(0, prefix)) == {1, ord(' ') + 2}
    with pytest.raises(ValueError, match='left the candidate'):
        allowed(0, torch.tensor([101, 102] + tokenizer.encode('INVALID')))
    with pytest.raises(ValueError, match='raise max_new_tokens'):
        _CandidateTrie(tokenizer, candidates, 1, 4)


@pytest.mark.parametrize(
    'candidates',
    [[], [CANDIDATES[0]] * 2, [dict(choice='blocked', instruction=LABELS[0])],
     [CANDIDATES[0],
      dict(choice='other', instruction=LABELS[0])]])
def test_bad_candidate_vocabulary_fails(candidates):
    with pytest.raises(ValueError):
        _CandidateTrie(Tokenizer(), candidates, 1, 128)


def test_subtask_checkpoint_cannot_enable_agent_completion(tmp_path):
    with pytest.raises(ValueError, match='cannot judge completion'):
        LocalVLMSubtaskPlanner(tmp_path / 'missing', completion_mode='agent')


def test_directory_resolution_uses_step_not_mtime(tmp_path):
    for name in ('step-9-epoch-1.safetensors', 'step-10-epoch-1.safetensors',
                 'step-20-epoch-1.safetensors.tmp'):
        (tmp_path / name).touch()
    assert resolve_planner_checkpoint(tmp_path).name.startswith('step-10-')
    (tmp_path / 'latest-checkpoint.safetensors').symlink_to(
        tmp_path / 'step-9-epoch-1.safetensors')
    assert resolve_planner_checkpoint(tmp_path).name.startswith('step-9-')
    with pytest.raises(FileNotFoundError):
        resolve_planner_checkpoint(tmp_path / 'missing')


def test_local_config_replaces_only_planner_and_keeps_full_schedule(
        monkeypatch):
    for key in ('PI05_ROBODOJO_CHECKPOINT_DIR', 'PI05_ROBODOJO_CHECKPOINT',
                'ROBODOJO_VLM_PLANNER_DIR', 'ROBODOJO_VLM_PLANNER_CHECKPOINT',
                'ROBODOJO_VLM_PLANNER_ASSETS', 'ROBODOJO_SUBTASK_DATA_ROOT'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('OPENAI_PLANNER_MODEL', 'must-not-be-used')
    monkeypatch.setenv('OPENAI_BASE_URL', 'must-not-be-used')
    cfg = Config.fromfile(ROOT /
                          'configs/hierarchical/robodojo_pi05_qwen35.py')
    baseline = Config.fromfile(ROOT /
                               'configs/hierarchical/robodojo_pi05_gpt6.py')
    assert cfg.policy.planner.type == 'LocalVLMSubtaskPlanner'
    assert not {'model', 'base_url', 'max_retries', 'allow_request_finish'} & \
        cfg.policy.planner.keys()
    assert cfg.model == baseline.model and cfg.themis == baseline.themis
    assert cfg.eval == baseline.eval
    assert cfg.policy.subtask_source.data_root == \
        baseline.policy.subtask_source.data_root
    assert cfg.policy.subtask_source.subtask_prefix == 'dense'
    assert cfg.policy.subtask_source.max_candidates == 4096
    assert baseline.policy.subtask_source.max_candidates == 4096
    assert cfg.policy.completion_mode == 'disabled'
    assert cfg.themis.server.runner.policy_execution == 'per_request'
    assert cfg.policy.planner.checkpoint_path.endswith(
        'qwen35_4b_robodojo_subtask_d78c5a778_4node_bs64_old_'
        '20261116_dlcf6osrr2ymq9yd/checkpoints/'
        'step-007311-epoch-03-loss=0.0678.safetensors')
    assert cfg.themis.server.ckpt_path.endswith(
        'pi05_robodojo_subtask_d78c5a778_8node_bs128_old_'
        '20261116_dlc1wn1n2jhosfyh/checkpoints/'
        'step-060000-epoch-04-loss=0.0011.safetensors')
    pi05_root = Path(cfg.themis.server.ckpt_path).parent.parent
    assert cfg.themis.server.norm_stats_path == str(pi05_root /
                                                    'dataset_statistics.json')
    assert cfg.model.pretrained_name_or_path == cfg.themis.server.ckpt_path
    prompt = next(t for t in cfg.eval.dataset.transforms
                  if t.type == 'ProcessPrompts')
    assert prompt.tokenizer.model_path == str(pi05_root / 'tokenizer')
    qwen_root = Path(cfg.policy.planner.checkpoint_path).parent.parent
    assert cfg.policy.planner.assets_path == str(qwen_root / 'vlm_assets')
    assert cfg.policy.planner.image_keys == cfg.policy.planner_image_keys


def test_local_paths_can_be_overridden_without_editing_config(monkeypatch):
    monkeypatch.setenv('PI05_ROBODOJO_CHECKPOINT_DIR', '/tmp/pi05-run')
    monkeypatch.setenv('PI05_ROBODOJO_CHECKPOINT', '/tmp/pi05.safetensors')
    monkeypatch.setenv('ROBODOJO_SUBTASK_DATA_ROOT', '/tmp/training-data')
    monkeypatch.setenv('ROBODOJO_VLM_PLANNER_DIR', '/tmp/finetuned')
    monkeypatch.setenv('ROBODOJO_VLM_PLANNER_CHECKPOINT',
                       '/tmp/another.safetensors')
    monkeypatch.setenv('ROBODOJO_VLM_PLANNER_ASSETS', '/tmp/saved-assets')
    monkeypatch.setenv('ROBODOJO_VLM_PLANNER_DEVICE', 'cuda:1')
    cfg = Config.fromfile(ROOT /
                          'configs/hierarchical/robodojo_pi05_qwen35.py')
    assert cfg.policy.planner.checkpoint_path == '/tmp/another.safetensors'
    assert cfg.policy.planner.assets_path == '/tmp/saved-assets'
    assert cfg.policy.planner.device == 'cuda:1'
    assert cfg.themis.server.ckpt_path == '/tmp/pi05.safetensors'
    assert cfg.model.pretrained_name_or_path == '/tmp/pi05.safetensors'
    assert cfg.themis.server.norm_stats_path == \
        '/tmp/pi05-run/dataset_statistics.json'
    assert cfg.policy.subtask_source.data_root == '/tmp/training-data'


def test_strict_loading_uses_saved_architecture_without_base_weights(
        tmp_path, monkeypatch):
    import fluxvla.engines.utils.builder as builder
    checkpoint = tmp_path / 'checkpoints/step-1-epoch-1.safetensors'
    checkpoint.parent.mkdir()
    assets = tmp_path / 'vlm_assets'
    assets.mkdir()
    architecture = dict(model_type='qwen3_5', test_saved_architecture=True)
    (assets / 'config.json').write_text(json.dumps(architecture))
    model = torch.nn.Linear(2, 2)
    state = {k: torch.full_like(v, 3) for k, v in model.state_dict().items()}
    save_file(state, str(checkpoint))
    build = Mock(return_value=model)
    monkeypatch.setattr(builder, 'build_vla_from_cfg', build)
    monkeypatch.setattr(vlm_transforms, 'get_processor_for_vlm',
                        Mock(return_value=Processor()))
    planner = LocalVLMSubtaskPlanner(
        checkpoint, torch_dtype='fp32', device='cpu')
    assert build.call_args.args[0]['vlm_backbone'][
        'model_config'] == architecture
    assert 'model_path' not in build.call_args.args[0]['vlm_backbone']
    assert not planner.model.training
    for name, param in planner.model.named_parameters():
        assert not param.requires_grad
        torch.testing.assert_close(param, state[name])
    save_file({'unexpected': torch.zeros(1)}, str(checkpoint))
    with pytest.raises(RuntimeError, match='Missing key'):
        LocalVLMSubtaskPlanner(checkpoint, torch_dtype='fp32', device='cpu')


def test_local_planner_through_existing_action_only_serving(local, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from fluxvla.engines.runners.serving.serving_runner import \
        FluxVLAServingRunner

    planner, _, model, _, _ = local
    root = tmp_path / 'data/meta/episodes'
    root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([
            dict(
                episode_index=0,
                length=2,
                tasks=['Stack the blocks.'],
                dense_subtask_names=LABELS,
                dense_subtask_start_frames=[0, 1],
                dense_subtask_end_frames=[1, 2])
        ]), root / 'episode.parquet')
    prompts = []

    def preprocess(observation):
        prompts.append(observation['task_description'])
        return {}

    class VLA(torch.nn.Module):

        def predict_action(self, **inputs):
            return torch.ones(1, 2, 14)

    runner = FluxVLAServingRunner(
        VLA(),
        dataset=preprocess,
        device='cpu',
        enable_mixed_precision=False,
        model_outputs_environment_actions=True,
        policy_execution='per_request',
        policy=dict(
            type='HierarchicalPolicy',
            planner=planner,
            subtask_source=dict(data_root=str(root.parents[1])),
            completion_mode='disabled',
            planner_image_keys=CAMERAS,
            monitor=None,
            goal_conditions=None))
    for _ in range(2):
        obs = request()['observation']
        obs['task_description'] = request()['goal']
        actions, _ = runner.predict_action(obs, '', policy_seed=42)
        assert actions.shape == (2, 14) and np.isfinite(actions).all()
        assert runner.last_prediction_metadata['subtask'][
            'instruction'] == LABELS[1]
        assert obs['task_description'] == 'Stack the blocks.'
        assert runner.policy._pending_horizon is None
        assert runner.policy.controller.history == []
    assert prompts == [LABELS[1], LABELS[1]]
    assert len(
        model.calls) == 2  # Same loaded VLM, two independent observations.
