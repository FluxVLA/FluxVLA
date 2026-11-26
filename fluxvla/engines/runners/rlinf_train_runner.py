"""FluxVLA-native entry point for RLinf training and evaluation."""

import copy
import importlib.util
import os
from pathlib import Path

from ..utils.root import RUNNERS


def _active_env_phases(cfg):
    if not cfg.runner.only_eval:
        yield 'train'
    if cfg.runner.only_eval or cfg.runner.get('val_check_interval', -1) > 0:
        yield 'eval'


def _positive_integer(field, value):
    if type(value) is not int or value < 1:
        raise ValueError(f'{field}={value!r} must be a positive integer')


def _validate_smolvla_batch_parameters(cfg):
    # Check divisors before RLinf uses them in its own placement checks.
    _positive_integer('rollout.pipeline_stage_num',
                      cfg.rollout.pipeline_stage_num)
    _positive_integer('actor.model.num_action_chunks',
                      cfg.actor.model.num_action_chunks)
    # RLinf checks actor batches for embodied tasks even in eval mode.
    if cfg.runner.task_type != 'embodied_eval':
        for field in ('micro_batch_size', 'global_batch_size'):
            _positive_integer(f'actor.{field}', cfg.actor[field])
    micro = cfg.actor.model.fluxvla.rollout_micro_batch_size
    if micro is not None:
        _positive_integer('actor.model.fluxvla.rollout_micro_batch_size',
                          micro)
    for phase in _active_env_phases(cfg):
        env = cfg.env[phase]
        for field in ('total_num_envs', 'max_steps_per_rollout_epoch'):
            _positive_integer(f'env.{phase}.{field}', env[field])
        _positive_integer(f'env.{phase}.group_size', env.get('group_size', 1))


def _validate_smolvla_batch_placement(cfg, placement):
    env_ranks = placement.get_world_size('env')
    rollout_ranks = placement.get_world_size('rollout')
    stages = cfg.rollout.pipeline_stage_num
    for phase in _active_env_phases(cfg):
        env = cfg.env[phase]
        _positive_integer(f'env.{phase}.rollout_epoch', env.rollout_epoch)
        if env.total_num_envs % (env_ranks * stages):
            raise ValueError(
                f'env.{phase}.total_num_envs={env.total_num_envs} must be '
                f'divisible by env ranks ({env_ranks}) × '
                f'pipeline stages ({stages})')
        # The synchronous RLinf router splits this stage batch at both ends.
        stage_batch = env.total_num_envs // stages
        if stage_batch % rollout_ranks:
            raise ValueError(
                f'env.{phase}.total_num_envs={env.total_num_envs} across '
                f'{stages} pipeline stages gives {stage_batch} envs '
                'per stage, '
                f'which must be divisible by rollout ranks ({rollout_ranks})')
    if cfg.runner.only_eval:
        return

    train = cfg.env.train
    actor_ranks = placement.get_world_size('actor')
    # RLinf splits each env-stage batch along the env axis before it
    # flattens time. Equal shards give every synchronous actor equal work.
    if train.total_num_envs % actor_ranks:
        raise ValueError(
            f'env.train.total_num_envs={train.total_num_envs} must be '
            f'divisible by actor ranks ({actor_ranks}) for equal '
            'trajectory shards')
    chunk_steps = (
        train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks)
    envs_per_actor = train.total_num_envs // actor_ranks
    samples_per_rank = envs_per_actor * train.rollout_epoch * chunk_steps
    batch_per_rank = cfg.actor.global_batch_size // actor_ranks
    if samples_per_rank % batch_per_rank:
        raise ValueError(
            f'Each of {actor_ranks} actor ranks receives {samples_per_rank} '
            f'rollout samples ({envs_per_actor} envs × '
            f'{train.rollout_epoch} rollout epochs × {chunk_steps} '
            'chunk steps), '
            'which must be divisible by the per-rank PPO batch '
            f'{batch_per_rank} '
            f'(actor.global_batch_size={cfg.actor.global_batch_size})')


@RUNNERS.register_module()
class RLinfTrainRunner:
    """Build and run RLinf workers through the native runner registry."""

    def __init__(self, cfg, mode='train'):
        if mode not in ('train', 'eval'):
            raise ValueError(f'Unsupported RLinf mode: {mode}')
        self.cfg = cfg
        self.mode = mode

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
        os.environ['RLINF_ROOT'] = str(rlinf_root)
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
                if (not evaluation and fsdp.strategy == 'fsdp2'
                        and fsdp.get('enable_gradient_accumulation', True)):
                    raise ValueError(
                        'PI05 RoboTwin requires '
                        'actor.fsdp_config.enable_gradient_accumulation=false '
                        'to preserve rollout/actor probability consistency')
                if (cfg.actor.model.fluxvla.rollout_micro_batch_size !=
                        cfg.actor.micro_batch_size):
                    raise ValueError(
                        'RoboTwin rollout and actor microbatches must match')
            elif (env.env_type != 'libero'
                  or env.task_suite_name != 'libero_10'):
                raise ValueError('Expected LIBERO-10 or RoboTwin')
        if not Path(cfg.actor.model.model_path).expanduser().exists():
            raise FileNotFoundError('Initial SFT checkpoint does not exist')
        if model_type == 'fluxvla_smolvla':
            from fluxvla.models.vlas.smolvla_rl_builder import \
                resolve_sft_artifacts
            options = cfg.actor.model.fluxvla
            resolve_sft_artifacts(
                cfg.actor.model.model_path,
                norm_stats_path=options.get('norm_stats_path'),
                tokenizer_path=options.get('tokenizer_path'))
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
        self.validate_frontend_cfg(self.cfg, evaluation=evaluation)
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
        actor_cfg = self.cfg.actor
        smolvla = actor_cfg.model.model_type == 'fluxvla_smolvla'
        if smolvla:
            options = actor_cfg.model.fluxvla
            if not evaluation and options.rollout_micro_batch_size is None:
                options.rollout_micro_batch_size = actor_cfg.micro_batch_size
            _validate_smolvla_batch_parameters(self.cfg)
        cfg = validate_cfg(self.cfg)
        cluster = Cluster(
            cluster_cfg=cfg.cluster,
            distributed_log_dir=cfg.runner.per_worker_log_path)
        placement = HybridComponentPlacement(cfg, cluster)
        if smolvla:
            _validate_smolvla_batch_placement(cfg, placement)
        workers = [('rollout', FluxRolloutWorker), ('env', FluxEnvWorker)]
        if not evaluation:
            from rlinf.workers.actor.embodied_fsdp_actor_worker import \
                EmbodiedFSDPActor
            if cfg.actor.model.model_type == 'fluxvla_smolvla':
                from .rlinf.smolvla_actor import SmolVLAActor
                actor = SmolVLAActor
            else:
                actor = EmbodiedFSDPActor
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
        else:
            from .rlinf_checkpoint_runner import FluxEmbodiedRunner
            runner = FluxEmbodiedRunner(cfg=cfg, **groups)
        runner.init_workers()
        return runner.run()
