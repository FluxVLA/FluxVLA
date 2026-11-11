"""Conservative PI05 training with checkpointed process restarts."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from supervise import supervise

CODE = Path(__file__).resolve().parents[3]
ROOT = BASE = WEIGHTS = RESULTS = None
EXPERIMENT = 'pi05-adjust-bottle-conservative'


def command(mode, end_step=1, resume=None):
    return [
        sys.executable,
        '-u',
        f'scripts/rl_{"eval" if mode == "baseline" else "train"}.py',
        'model=pi05',
        'simulator=robotwin',
        f'actor.model.model_path={WEIGHTS}',
        f'env.train.assets_path={BASE}',
        'env.train.task_config.task_name=adjust_bottle',
        'env.train.task_config.step_lim=200',
        'cluster.component_placement={actor: 0-7, env: 0-7, rollout: 0-7}',
        'actor.micro_batch_size=8',
        'actor.global_batch_size=512',
        'actor.model.fluxvla.rollout_micro_batch_size=8',
        'actor.optim.lr=5e-6',
        'actor.optim.value_lr=1e-4',
        'actor.optim.total_training_steps=1000',
        'actor.seed=1234',
        'algorithm.update_epoch=5',
        'actor.enable_offload=true',
        'rollout.enable_offload=true',
        'env.enable_offload=true',
        'env.train.enable_offload=true',
        'env.eval.enable_offload=true',
        'env.train.total_num_envs=64',
        'env.train.rollout_epoch=4',
        'env.train.seed=0',
        'env.train.auto_reset=false',
        'env.train.ignore_terminations=false',
        'env.eval.total_num_envs=128',
        'env.eval.rollout_epoch=1',
        'env.eval.seed=0',
        'env.eval.use_fixed_reset_state_ids=true',
        f'runner.max_steps={end_step}',
        'runner.max_epochs=1000',
        'runner.val_check_interval=5',
        'runner.save_interval=5',
        'runner.logger.logger_backends=[wandb,tensorboard]',
        'runner.logger.project_name=fluxvla-rl-robotwin',
        f'runner.logger.log_path={RESULTS / mode}',
        f'runner.logger.experiment_name={EXPERIMENT}-{mode}',
    ] + ([f'runner.resume_dir={resume}'] if resume else [])


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def checkpoint_path(step):
    return (RESULTS / 'train' / f'{EXPERIMENT}-train' / 'checkpoints' /
            f'global_step_{step}')


def validate_checkpoint(path):
    actor = path / 'actor'
    weights = actor / 'model_state_dict/full_weights.pt'
    metadata = actor / 'dcp_checkpoint/.metadata'
    shards = list((actor / 'dcp_checkpoint').glob('*.distcp'))
    if not weights.is_file() or not weights.stat().st_size:
        raise RuntimeError(f'Missing full weights: {weights}')
    if not metadata.is_file() or len(shards) < 8 or any(not p.stat().st_size
                                                        for p in shards):
        raise RuntimeError(
            f'Incomplete distributed training checkpoint: {actor}')


def campaign():
    (ROOT / 'commands').mkdir(exist_ok=True)
    (ROOT / 'logs').mkdir(exist_ok=True)
    ledger_path = ROOT / 'checkpoints.json'
    ledger = json.loads(
        ledger_path.read_text()) if ledger_path.exists() else []
    start_step = ledger[-1]['step'] if ledger else 0
    for end_step in [1] + list(range(5, 1001, 5)):
        if end_step <= start_step:
            continue
        resume = checkpoint_path(start_step) if start_step else None
        if resume:
            validate_checkpoint(resume)
        if shutil.disk_usage(ROOT).free < 100 * 1024**3:
            raise RuntimeError('Less than 100 GiB disk space remains')
        cmd = command('train', end_step, resume)
        write_json(ROOT / 'commands' / f'train_to_{end_step}.json', cmd)
        env = os.environ.copy()
        env['WANDB_RUN_ID'] = f'{ROOT.name}-to{end_step}-{time.time_ns()}'
        env.setdefault('WANDB_RUN_GROUP', ROOT.name)
        env.setdefault('WANDB_MODE', 'offline')
        env['WANDB_JOB_TYPE'] = 'ppo-train-segment'
        env['RAY_TMPDIR'] = tempfile.mkdtemp(prefix=f'ray-pi05-to{end_step}-')
        logfile = ROOT / 'logs' / f'train_to_{end_step}_{time.time_ns()}.log'
        with logfile.open('a') as log:
            log.write(f'\nStarting segment {start_step} -> {end_step}; '
                      f'resume={resume}\n')
            log.flush()
            process = subprocess.Popen(
                cmd,
                cwd=CODE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True)
            status = {
                'phase': 'train',
                'state': 'running',
                'pid': process.pid,
                'started_at': time.time(),
                'segment_start_step': start_step,
                'segment_end_step': end_step,
                'target_step': 1000,
                'resume_dir': str(resume) if resume else None,
                'log_file': str(logfile),
                'wandb_mode': env.get('WANDB_MODE')
            }
            write_json(ROOT / 'status.json', status)
            code, failure = supervise(process, logfile, ROOT)
        if code or failure:
            status.update(
                state='failed',
                returncode=code,
                failure_reason=failure,
                finished_at=time.time())
            write_json(ROOT / 'status.json', status)
            raise SystemExit(code or 1)
        checkpoint = checkpoint_path(end_step)
        validate_checkpoint(checkpoint)
        ledger.append({
            'step': end_step,
            'path': str(checkpoint),
            'completed_at': time.time(),
            'resume_from': str(resume) if resume else None
        })
        write_json(ledger_path, ledger)
        start_step = end_step
        status.update(
            last_completed_checkpoint=str(checkpoint),
            completed_runner_steps=end_step)
        if end_step == 1000:
            status.update(state='completed', finished_at=time.time())
        write_json(ROOT / 'status.json', status)


def main():
    global ROOT, BASE, WEIGHTS, RESULTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--assets-path', type=Path, required=True)
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Initial SFT checkpoint, including tokenizer/stats')
    parser.add_argument(
        '--resolve',
        action='store_true',
        help='Write the resolved first-segment config and exit')
    args = parser.parse_args()
    ROOT = args.run_dir.expanduser().resolve()
    BASE = args.assets_path.expanduser().resolve()
    WEIGHTS = args.checkpoint.expanduser().resolve()
    RESULTS = ROOT / 'results'
    for path in (BASE, WEIGHTS):
        if not path.is_dir():
            parser.error(f'Directory does not exist: {path}')
    ROOT.mkdir(parents=True, exist_ok=True)
    try:
        if args.resolve:
            result = subprocess.run(
                command('train') + ['--cfg', 'job', '--resolve'],
                cwd=CODE,
                capture_output=True,
                text=True,
                check=True)
            (ROOT / 'train.resolve.log').write_text(result.stdout +
                                                    result.stderr)
            lines = result.stdout.splitlines(keepends=True)
            start = next(i for i, line in enumerate(lines)
                         if line == 'actor:\n')
            (ROOT / 'train.resolved.yaml').write_text(''.join(lines[start:]))
        else:
            campaign()
    except Exception as exc:
        path = ROOT / 'status.json'
        status = json.loads(path.read_text()) if path.exists() else {}
        status.update(
            state='failed',
            failure_reason=f'{type(exc).__name__}: {exc}',
            finished_at=time.time())
        write_json(path, status)
        raise


if __name__ == '__main__':
    main()
