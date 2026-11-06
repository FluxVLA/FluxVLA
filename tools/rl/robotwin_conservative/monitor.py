"""Read training events and keep a small reproduction progress report."""

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

# The event reader needs TensorBoard's filesystem stub, not TensorFlow/CUDA.
sys.modules['tensorboard.compat.notf'] = types.ModuleType(
    'tensorboard.compat.notf')

ROOT = RESULTS = None


def snapshot():
    from tensorboard.backend.event_processing.event_accumulator import \
        EventAccumulator

    status = json.loads((ROOT / 'status.json').read_text())
    metrics = {}
    evaluations = []
    updates = set()
    logdir = RESULTS / 'train' / 'tensorboard'
    if logdir.is_dir():
        events = EventAccumulator(str(logdir), size_guidance={'scalars': 0})
        events.Reload()
        for tag in events.Tags()['scalars']:
            values = events.Scalars(tag)
            if not values:
                continue
            last = values[-1]
            metrics[tag] = {'logged_step': last.step, 'value': last.value}
            if tag.startswith('train/'):
                updates.update(x.step for x in values)
            if tag == 'eval/success_once':
                evaluations = [{
                    'completed_runner_step': x.step + 1,
                    'success_rate': x.value
                } for x in values]
    nonfinite = [
        tag for tag, item in metrics.items()
        if not math.isfinite(item['value'])
    ]
    output = {
        'updated_at':
        time.time(),
        'process':
        status,
        'completed_runner_steps':
        max(updates) + 1 if updates else 0,
        'first_update_verified':
        bool(updates) and not nonfinite,
        'nonfinite_metric_keys':
        nonfinite,
        'latest_metrics':
        metrics,
        'evaluations':
        evaluations,
        'best_eval_success_rate':
        max((x['success_rate'] for x in evaluations), default=None),
        'wandb_mode':
        status.get('wandb_mode'),
    }
    path = ROOT / 'progress.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(output, indent=2, allow_nan=True) + '\n')
    temp.replace(path)
    return status['state']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    ROOT = args.run_dir.expanduser().resolve()
    RESULTS = ROOT / 'results'
    while True:
        try:
            state = snapshot()
            if state in ('failed', 'completed'):
                break
        except Exception as exc:
            print(type(exc).__name__, str(exc), flush=True)
        if args.once:
            break
        time.sleep(60)
