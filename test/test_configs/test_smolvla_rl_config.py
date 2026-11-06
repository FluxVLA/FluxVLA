"""RL config parsing must not change the legacy SFT statistics source."""

from pathlib import Path

from mmengine import Config

ROOT = Path(__file__).resolve().parents[2]


def test_smolvla_rl_config_parses_and_isolates_statistics():
    sft_path = ROOT / 'configs/smolvla/smolvla_libero_10_finetune.py'
    rl_path = ROOT / 'configs/rl/smolvla_libero_10.py'
    sft_before = Config.fromfile(sft_path)
    rl = Config.fromfile(rl_path)
    sft_after = Config.fromfile(sft_path)
    pi05 = Config.fromfile(
        ROOT / 'configs/pi05/pi05_paligemma_libero_10_full_finetune.py')

    assert rl.model == sft_before.model
    assert sft_before.train_dataloader == sft_after.train_dataloader
    assert 'dataset_statistics' not in sft_after.train_dataloader.dataset
    assert (rl.train_dataloader.dataset.dataset_statistics ==
            pi05.train_dataloader.dataset.dataset_statistics)
    stats = rl.train_dataloader.dataset.pop('dataset_statistics')
    assert rl.train_dataloader == sft_before.train_dataloader
    # Repeated loads must rebuild the RL override after callers mutate it.
    assert Config.fromfile(
        rl_path
    ).train_dataloader.dataset.dataset_statistics == stats  # noqa: E501
