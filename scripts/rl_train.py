"""Train a registered FluxVLA policy with RLinf."""

import torch.multiprocessing as mp

from fluxvla.engines import build_runner_from_cfg
from fluxvla.engines.runners.rlinf_train_runner import RLinfTrainRunner


def main():
    root = RLinfTrainRunner.prepare_environment()
    import hydra

    @hydra.main(
        version_base='1.3',
        config_path=str(root / 'configs/rl'),
        config_name='default')
    def launch(cfg):
        runner = build_runner_from_cfg({
            'type': 'RLinfTrainRunner',
            'cfg': cfg,
            'mode': 'train'
        })
        runner.run()

    mp.set_start_method('spawn', force=True)
    launch()


if __name__ == '__main__':
    main()
