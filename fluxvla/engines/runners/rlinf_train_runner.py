"""FluxVLA-native entry point for RLinf train, evaluation and preflight."""

import copy
import importlib.util
import os
from pathlib import Path

from ..utils.root import RUNNERS


@RUNNERS.register_module()
class RLinfTrainRunner:
    """Build and run RLinf workers through the native runner registry."""

    def __init__(self, cfg, mode='train', preflight=False):
        if mode not in ('train', 'eval'):
            raise ValueError(f'Unsupported RLinf mode: {mode}')
        self.cfg = cfg
        self.mode = mode
        self.preflight = bool(preflight)

    @staticmethod
    def prepare_environment():
        flux_root = Path(__file__).resolve().parents[3]
        spec = importlib.util.find_spec('rlinf')
        if spec is None or spec.origin is None:
            raise ImportError(
                'RLinf is not installed; set RLINF_ROOT and install editable')
        rlinf_root = Path(spec.origin).resolve().parents[1]
        config_dir = Path(
            os.environ.get('RLINF_CONFIG_DIR',
                           rlinf_root / 'examples/embodiment/config'))
        if not config_dir.is_dir():
            raise FileNotFoundError(
                f'RLinf configs missing: {config_dir}. Set RLINF_CONFIG_DIR')
        os.environ['FLUXVLA_ROOT'] = str(flux_root)
        os.environ['RLINF_CONFIG_DIR'] = str(config_dir.resolve())
        extension = 'fluxvla.engines.runners.rlinf.registry'
        existing = os.environ.get('RLINF_EXT_MODULE')
        if existing and existing != extension:
            raise ValueError(
                f'Conflicting RLINF_EXT_MODULE={existing}. Expected '
                f'{extension}')
        os.environ['RLINF_EXT_MODULE'] = extension
        os.environ['FLUX_RL_EXTERNAL_DISTRIBUTED'] = '1'
        # Hydra discovers YAML files automatically; register Python defaults
        # before either entry point composes the configuration.
        from hydra.core.config_store import ConfigStore
        from mmengine import Config

        config_root = flux_root / 'configs/rl'
        store = ConfigStore.instance()
        store.store(
            name='default',
            node=Config.fromfile(str(config_root / 'default.py')).to_dict())
        for path in sorted((config_root / 'model').glob('*.py')):
            store.store(
                group='model',
                name=path.stem,
                package='actor.model',
                node=Config.fromfile(str(path)).model.to_dict())
        for path in sorted((config_root / 'simulator').glob('*.py')):
            store.store(
                group='simulator',
                name=path.stem,
                package='_global_',
                node=Config.fromfile(str(path)).to_dict())
        return flux_root

    @staticmethod
    def validate_frontend_cfg(cfg, *, evaluation=False):
        if (cfg.algorithm.loss_type != 'actor_critic'
                or cfg.algorithm.adv_type != 'gae'):
            raise ValueError('FluxVLA RL supports PPO actor_critic + gae only')
        if cfg.algorithm.entropy_bonus != 0:
            raise ValueError('FluxVLA PPO requires entropy_bonus=0')
        if (cfg.actor.training_backend != 'fsdp'
                or cfg.actor.model.joint_logprob):
            raise ValueError(
                'FluxVLA PPO requires FSDP and joint_logprob=False')
        if (cfg.runner.get('use_training_pipeline', False)
                or cfg.runner.get('enable_decoupled_mode', False)):
            raise ValueError('FluxVLA RL uses the synchronous embodied runner')
        if cfg.runner.get('only_eval', False) != evaluation:
            expected = 'true' if evaluation else 'false'
            raise ValueError(f'runner.only_eval must be {expected}')
        model_type = cfg.actor.model.model_type
        supported_models = {'fluxvla_pi05', 'fluxvla_smolvla'}
        if model_type not in supported_models:
            raise ValueError(
                f'Unsupported actor.model.model_type={model_type}')
        if (cfg.actor.model.precision != 'fp32'
                or cfg.rollout.model.precision != 'fp32'):
            raise ValueError('Actor and rollout master weights must stay FP32')
        fsdp = cfg.actor.fsdp_config
        if fsdp.mixed_precision.param_dtype != 'fp32':
            raise ValueError('FSDP param_dtype must stay fp32')
        if (cfg.actor.model.fluxvla.compute_dtype == 'bf16'
                and (fsdp.strategy != 'fsdp2'
                     or fsdp.mixed_precision.cast_forward_inputs)):
            raise ValueError(
                'BF16 requires FSDP2 with cast_forward_inputs=False')
        if (cfg.actor.get('enable_sft_co_train', False)
                or cfg.actor.model.get('is_lora', False)):
            raise ValueError('Mixed SFT and LoRA are not supported')
        if fsdp.gradient_checkpointing:
            raise ValueError('Gradient checkpointing is not supported')
        if (cfg.rollout.get('enable_torch_compile', False)
                or cfg.rollout.get('enable_cuda_graph', False)):
            raise ValueError('Compile and CUDA graph are not supported')
        for phase in ('train', 'eval'):
            env = cfg.env[phase]
            if env.env_type == 'robotwin':
                if model_type == 'fluxvla_smolvla':
                    raise ValueError(
                        'SmolVLA RL currently supports LIBERO-10 only; its '
                        'RoboTwin preprocessing/action contract is not yet '
                        'implemented')
                if (cfg.actor.model.fluxvla.rollout_micro_batch_size !=
                        cfg.actor.micro_batch_size):
                    raise ValueError(
                        'RoboTwin rollout and actor microbatches must match')
                if env.task_config.task_name != 'adjust_bottle':
                    raise ValueError(
                        'Only RoboTwin adjust_bottle is supported')
            elif (env.env_type != 'libero'
                  or env.task_suite_name != 'libero_10'):
                raise ValueError(
                    'Expected LIBERO-10 or RoboTwin adjust_bottle')
        if not Path(cfg.actor.model.model_path).expanduser().exists():
            raise FileNotFoundError('Initial SFT checkpoint does not exist')
        if cfg.runner.resume_dir and not Path(
                cfg.runner.resume_dir).expanduser().is_dir():
            raise FileNotFoundError(
                f'Resume checkpoint does not exist: {cfg.runner.resume_dir}')

    def run(self):
        self.prepare_environment()
        from rlinf.config import validate_cfg
        from rlinf.scheduler import Cluster
        from rlinf.utils.placement import HybridComponentPlacement

        from .rlinf import FluxEnvWorker, FluxRolloutWorker
        from .rlinf.registry import register

        register()
        evaluation = self.mode == 'eval'
        self.cfg.runner.only_eval = evaluation
        if evaluation:
            self.cfg.rollout.model = copy.deepcopy(self.cfg.actor.model)
        if evaluation and self.cfg.simulator.name == 'robotwin_adjust_bottle':
            self.cfg.experiment.split = 'test'
            self.cfg.env.eval.total_num_envs = 10
            self.cfg.env.eval.rollout_epoch = 15
        self.validate_frontend_cfg(self.cfg, evaluation=evaluation)
        if self.preflight:
            if evaluation:
                raise ValueError('Preflight is a training-only mode')
            from .rlinf.preflight import run_gate
            return run_gate(self.cfg)
        if evaluation and self.cfg.runner.resume_dir:
            raise ValueError(
                'Evaluation uses runner.ckpt_path, not resume_dir')
        if evaluation and self.cfg.runner.ckpt_path:
            path = Path(self.cfg.runner.ckpt_path).expanduser().resolve()
            if path.is_dir():
                path = path / 'actor/model_state_dict/full_weights.pt'
            if not path.is_file():
                raise FileNotFoundError(f'RLinf weights missing: {path}')
            self.cfg.runner.ckpt_path = str(path)
        if 'wandb' in self.cfg.runner.logger.logger_backends:
            from .rlinf.experiment import require_wandb
            require_wandb()
        cfg = validate_cfg(self.cfg)
        cluster = Cluster(
            cluster_cfg=cfg.cluster,
            distributed_log_dir=cfg.runner.per_worker_log_path)
        placement = HybridComponentPlacement(cfg, cluster)
        workers = [('rollout', FluxRolloutWorker), ('env', FluxEnvWorker)]
        if not evaluation:
            from rlinf.workers.actor.embodied_fsdp_actor_worker import \
                EmbodiedFSDPActor
            actor = EmbodiedFSDPActor
            if cfg.env.train.env_type == 'robotwin':
                from .rlinf.actor_worker import AuditedEmbodiedActor
                actor = AuditedEmbodiedActor
            workers.insert(0, ('actor', actor))
        groups = {
            name: worker.create_group(cfg).launch(
                cluster,
                name=cfg[name].group_name,
                placement_strategy=placement.get_strategy(name))
            for name, worker in workers
        }
        if evaluation:
            from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
            runner = EmbodiedEvalRunner(cfg=cfg, **groups)
        elif cfg.env.train.env_type == 'robotwin':
            from .rlinf.pilot_runner import RoboTwinPilotRunner
            runner = RoboTwinPilotRunner(cfg=cfg, **groups)
        else:
            from rlinf.runners.embodied_runner import EmbodiedRunner
            runner = EmbodiedRunner(cfg=cfg, **groups)
        runner.init_workers()
        return runner.run()
