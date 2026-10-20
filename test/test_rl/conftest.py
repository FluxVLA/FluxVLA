"""Small, real SigLIP + two Gemma experts; no downloaded checkpoint needed."""

import io
from pathlib import Path

import numpy as np
import pytest
import sentencepiece as spm
import torch
from mmengine import Config
from omegaconf import OmegaConf

from fluxvla.engines import build_vla_from_cfg
from fluxvla.rl.models.builder import build_pi05_policy

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='session', autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope='session')
def tiny_assets(tmp_path_factory):
    root = tmp_path_factory.mktemp('flux_pi05_rl')
    buffer = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter([
            'pick the red cube', 'place the blue cup', 'open the drawer',
            'close the door'
        ] * 20),
        model_writer=buffer,
        vocab_size=40,
        hard_vocab_limit=False,
        minloglevel=2)
    tokenizer = root / 'tokenizer.model'
    tokenizer.write_bytes(buffer.getvalue())
    cfg = Config.fromfile(
        str(ROOT / 'configs/pi05/pi05_paligemma_libero_10_full_finetune.py'))
    model = cfg.model
    model.pretrained_name_or_path = None
    model.name_mapping = None
    model.proj_width = 16
    model.n_action_steps = 3
    model.num_steps = 3
    model.enable_mixed_precision_training = False
    for key, hidden in (('llm_backbone', 32), ('llm_expert', 16)):
        block = model[key]
        block.hidden_size = hidden
        block.intermediate_size = hidden * 2
        block.head_dim = 8
        block.num_attention_heads = 2
        block.num_key_value_heads = 1
        block.num_hidden_layers = 2
        block.vocab_size = 64
        block.adarms_cond_dim = None if key == 'llm_backbone' else 16
    vision = model.vision_backbone.vision_config
    vision.hidden_size = 16
    vision.intermediate_size = 32
    vision.num_attention_heads = 2
    vision.num_hidden_layers = 1
    vision.image_size = 16
    vision.patch_size = 8
    vision.projection_dim = 32
    model.projector.in_dim, model.projector.out_dim = 16, 32
    model.action_in_proj.out_dim = 16
    model.action_out_proj.in_dim = 16
    for key in ('time_mlp_in', 'time_mlp_out'):
        model[key].in_dim = model[key].out_dim = 16
    cfg.eval.dataset.transforms[1].input_sizes = [[3, 16, 16], [3, 16, 16]]
    cfg.eval.dataset.transforms[2].max_len = 12
    cfg.eval.dataset.transforms[2].tokenizer.model_path = str(tokenizer)
    config_path = root / 'tiny_pi05.py'
    cfg.dump(str(config_path))
    torch.manual_seed(10)
    original = build_vla_from_cfg(cfg.model)
    checkpoint = root / 'sft.pt'
    torch.save({'model': original.state_dict()}, checkpoint)
    return cfg, config_path, checkpoint, tokenizer


@pytest.fixture
def model_cfg(tiny_assets):
    _, config, checkpoint, tokenizer = tiny_assets
    return OmegaConf.create({
        'model_type': 'fluxvla_pi05',
        'model_path': str(checkpoint),
        'precision': 'fp32',
        'load_to_device': False,
        'is_lora': False,
        'add_value_head': True,
        'num_action_chunks': 2,
        'action_dim': 7,
        'num_steps': 3,
        'joint_logprob': False,
        'fluxvla': {
            'config_path': str(config),
            'tokenizer_path': str(tokenizer),
            'compute_dtype': 'fp32',
            'noise_level': 0.5,
            'train_expert_only': True,
            'noise_method': 'flow_sde'
        }
    })


@pytest.fixture
def policy(model_cfg):
    torch.manual_seed(5)
    return build_pi05_policy(model_cfg)


@pytest.fixture
def env_obs():
    rng = np.random.default_rng(42)
    return {
        'main_images':
        torch.from_numpy(rng.integers(0, 256, (3, 20, 24, 3), dtype=np.uint8)),
        'wrist_images':
        torch.from_numpy(rng.integers(0, 256, (3, 20, 24, 3), dtype=np.uint8)),
        'states':
        torch.from_numpy(rng.normal(size=(3, 8)).astype(np.float32)),
        'task_descriptions':
        ['pick the red cube', 'place the blue cup', 'open the drawer']
    }
