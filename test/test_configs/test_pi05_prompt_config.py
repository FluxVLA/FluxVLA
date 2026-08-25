# Copyright 2026 Limx Dynamics

from mmengine import Config

FLUXBISIM_CONFIGS = [
    'configs/pi05/fluxbisim/pi05_paligemma_close_box_full_finetune.py',
    'configs/pi05/fluxbisim/pi05_paligemma_handover_book_full_finetune.py',
    'configs/pi05/fluxbisim/pi05_paligemma_pick_place_banana_full_finetune.py',
    'configs/pi05/fluxbisim/pi05_paligemma_pull_push_drawer_full_finetune.py',
    'configs/pi05/fluxbisim/pi05_paligemma_screw_pitcher_lid_full_finetune.py',
]


def _get_transform(transforms, transform_type):
    return next(transform for transform in transforms
                if transform.type == transform_type)


def _assert_aloha_prompt_contract(transforms):
    prepare = _get_transform(transforms, 'PreparePromptWithState')
    process = _get_transform(transforms, 'ProcessPrompts')

    assert prepare.token_state_dim == 14
    assert process.max_len == 200
    assert process.preserve_suffix_after == ', State: '


def test_pi05_aloha_train_and_eval_preserve_the_openpi_prompt_contract():
    cfg = Config.fromfile('configs/pi05/pi05_paligemma_aloha_full_finetune.py')

    train_transforms = cfg.train_dataloader.dataset.datasets[0].transforms
    _assert_aloha_prompt_contract(train_transforms)
    _assert_aloha_prompt_contract(cfg.inference.dataset.transforms)


def test_pi05_aloha_rtc_train_and_inference_use_the_same_prompt_contract():
    cfg = Config.fromfile(
        'configs/pi05/pi05_paligemma_aloha_rtc_kernel_inference.py')

    train_transforms = cfg.train_dataloader.dataset.datasets[0].transforms
    _assert_aloha_prompt_contract(train_transforms)
    _assert_aloha_prompt_contract(cfg.inference.dataset.transforms)


def test_pi05_mixed_training_keeps_the_aloha_state_at_14_dimensions():
    cfg = Config.fromfile(
        'configs/pi05/pi05_paligemma_aloha+ur_4090_full_finetune.py')

    transforms = cfg.train_dataloader.dataset.datasets.aloha[0].transforms
    _assert_aloha_prompt_contract(transforms)


def test_pi05_fluxbisim_configs_use_the_14d_aloha_prompt_contract():
    for config_path in FLUXBISIM_CONFIGS:
        cfg = Config.fromfile(config_path)
        train_transforms = cfg.train_dataloader.dataset.datasets[0].transforms
        _assert_aloha_prompt_contract(train_transforms)
        _assert_aloha_prompt_contract(cfg.inference.dataset.transforms)
