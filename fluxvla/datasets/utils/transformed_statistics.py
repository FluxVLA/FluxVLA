#!/usr/bin/env python3
"""Compute transformed state/action statistics from LeRobot parquet data.

Robot values are first represented in the configured model action space,
selected absolute action dimensions are converted to deltas from the current
state, and statistics are computed only then. Normalization itself is
deliberately not applied here. The implementation is model-agnostic; profiles
describe robot/action-space preprocessing rather than a model family.

The default output is a Python literal that can be pasted directly into a
FluxVLA config as ``dataset_statistics``.  JSON output is also available for
inspection and tooling.
"""

from __future__ import annotations
import argparse
import json
import pprint
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class Profile:
    """Robot action-space fields required before normalization."""

    state_key: str
    action_key: str
    delta_mask: tuple[bool, ...]
    signs: tuple[float, ...] | None = None
    aloha_gripper_input_range: tuple[float, float] | None = None


PROFILES = {
    'absolute':
    Profile(state_key='observation.state', action_key='action', delta_mask=()),
    'aloha':
    Profile(
        state_key='observation.state',
        action_key='action',
        delta_mask=(True, ) * 6 + (False, ) + (True, ) * 6 + (False, ),
        signs=(1, -1, -1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1),
        aloha_gripper_input_range=(-0.01, 0.08)),
    'ur3':
    Profile(
        state_key='observation.state',
        action_key='action',
        delta_mask=(True, ) * 6 + (False, )),
    'franka-qpos':
    Profile(
        state_key='observation.state',
        action_key='observation.state',
        delta_mask=(True, ) * 7 + (False, ) + (True, ) * 7 + (False, )),
    'franka-eepose':
    Profile(
        state_key='observation.eepose',
        action_key='observation.eepose',
        delta_mask=()),
    'tron2':
    Profile(
        state_key='observation.state',
        action_key='action',
        # The restored Tron2 recipe predicts absolute arm/head/gripper qpos.
        delta_mask=()),
    'robocasa-joint-delta':
    Profile(
        state_key='observation.state',
        action_key='action',
        # GR1 arms and waist are absolute joint-position targets. Fourier
        # hand actions are discrete commands and must remain absolute.
        delta_mask=((True, ) * 7 + (False, ) * 6 + (True, ) * 7 +
                    (False, ) * 6 + (True, ) * 3)),
}


def _parse_bool_mask(value: str | None) -> tuple[bool, ...] | None:
    if value is None:
        return None
    items = [item.strip().lower() for item in value.split(',')]
    parsed = []
    for item in items:
        if item in {'1', 'true', 't', 'yes', 'y'}:
            parsed.append(True)
        elif item in {'0', 'false', 'f', 'no', 'n'}:
            parsed.append(False)
        else:
            raise argparse.ArgumentTypeError(
                'delta mask entries must be booleans or 0/1, got '
                f'{item!r}')
    if not parsed:
        raise argparse.ArgumentTypeError('delta mask must not be empty')
    return tuple(parsed)


def _discover_dataset_roots(paths: Sequence[Path]) -> list[Path]:
    roots = set()
    for path in paths:
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f'Dataset path does not exist: {path}')
        if (path / 'meta' / 'info.json').is_file() and (path /
                                                        'data').is_dir():
            roots.add(path)
            continue
        for info_path in path.rglob('meta/info.json'):
            root = info_path.parent.parent
            if (root / 'data').is_dir():
                roots.add(root)
    if not roots:
        joined = ', '.join(str(path) for path in paths)
        raise ValueError(
            'No LeRobot dataset roots were found. Expected each input or one '
            f'of its children to contain meta/info.json and data/: {joined}')
    return sorted(roots)


def _parquet_files(roots: Sequence[Path]) -> list[Path]:
    files = sorted({
        file
        for root in roots for file in (root / 'data').rglob('*.parquet')
    })
    if not files:
        raise ValueError(
            'No parquet files found under the discovered data/ directories.')
    return files


def _column_to_numpy(table, key: str) -> np.ndarray:
    if key not in table.column_names:
        raise KeyError(f'Column {key!r} is missing. Available columns: '
                       f'{table.column_names}')
    values = np.asarray(table[key].to_pylist(), dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(
            f'Column {key!r} must contain vectors, got shape {values.shape}.')
    return values


def _dataset_root_for_parquet(file_path: Path) -> Path:
    for parent in file_path.parents:
        if parent.name == 'data':
            return parent.parent
    raise ValueError(f'Parquet file is not under a data/ directory: '
                     f'{file_path}')


def _iter_episodes(files: Sequence[Path], state_key: str,
                   action_key: str) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ordered state/action arrays one complete episode at a time."""
    seen_episodes: set[tuple[Path, int]] = set()
    for file_path in files:
        schema_names = set(pq.ParquetFile(file_path).schema_arrow.names)
        order_key = ('frame_index' if 'frame_index' in schema_names else
                     'timestamp' if 'timestamp' in schema_names else None)
        columns = list(
            dict.fromkeys([
                state_key,
                action_key,
                'episode_index' if 'episode_index' in schema_names else None,
                order_key,
            ]))
        columns = [column for column in columns if column is not None]
        table = pq.read_table(file_path, columns=columns)
        states = _column_to_numpy(table, state_key)
        actions = _column_to_numpy(table, action_key)
        if len(states) != len(actions):
            raise ValueError(f'State/action row mismatch in {file_path}: '
                             f'{len(states)} != {len(actions)}')

        if 'episode_index' in table.column_names:
            episode_indices = np.asarray(table['episode_index'].to_pylist())
        else:
            episode_indices = np.zeros(len(states), dtype=np.int64)
        if order_key is not None:
            order_values = np.asarray(table[order_key].to_pylist())
        else:
            order_values = np.arange(len(states))

        for episode_index in np.unique(episode_indices):
            selector = np.flatnonzero(episode_indices == episode_index)
            selector = selector[np.argsort(
                order_values[selector], kind='stable')]
            # LeRobot v2 normally stores one episode per file. Detect a split
            # episode instead of silently computing incorrect action windows.
            episode_id = (_dataset_root_for_parquet(file_path),
                          int(episode_index))
            if episode_id in seen_episodes:
                raise ValueError(
                    f'Episode {episode_index} is split across parquet files '
                    f'under {episode_id[0]}; merge it before computing stats.')
            seen_episodes.add(episode_id)
            yield states[selector], actions[selector]


def _infer_profile(state_dim: int, state_key: str, action_key: str) -> Profile:
    if state_key == 'observation.eepose':
        if state_dim != 16:
            raise ValueError(
                'Automatic Franka eepose inference expects 16 dimensions, '
                f'got {state_dim}. Pass --profile or explicit keys/mask.')
        return PROFILES['franka-eepose']
    if state_dim == 7:
        return PROFILES['ur3']
    if state_dim == 16 and action_key == state_key:
        return PROFILES['franka-qpos']
    if state_dim == 18:
        return PROFILES['tron2']
    raise ValueError(
        f'Cannot infer a safe delta mask for {state_dim}D {state_key!r}/'
        f'{action_key!r}. Use --profile or --delta-mask explicitly.')


def _apply_signs(values: np.ndarray,
                 signs: tuple[float, ...] | None) -> np.ndarray:
    if signs is None:
        return values
    if values.shape[-1] != len(signs):
        raise ValueError(f'Expected {len(signs)} dimensions for signs, got '
                         f'{values.shape[-1]}.')
    return values * np.asarray(signs, dtype=np.float32)


def _normalize_range(values: np.ndarray, low: float,
                     high: float) -> np.ndarray:
    return (values - low) / (high - low)


def _unnormalize_range(values: np.ndarray, low: float,
                       high: float) -> np.ndarray:
    return values * (high - low) + low


def _aloha_gripper_to_angular(values: np.ndarray) -> np.ndarray:
    values = _unnormalize_range(values, 0.01844, 0.05800)
    arm_length = 0.036
    horn_radius = 0.022
    argument = (horn_radius**2 + values**2 -
                arm_length**2) / (2 * horn_radius * values)
    values = np.arcsin(np.clip(argument, -1.0, 1.0))
    return _normalize_range(values, 0.5476, 1.6296)


def _aloha_gripper_from_angular_inv(values: np.ndarray) -> np.ndarray:
    values = _unnormalize_range(values, -0.6213, 1.4910)
    return values - 0.5476


def _apply_aloha_coordinates(
    states: np.ndarray, actions: np.ndarray,
    gripper_input_range: tuple[float, float] | None
) -> tuple[np.ndarray, np.ndarray]:
    """Match ``OpenPIAlohaGripperCoordinates`` before delta conversion."""
    states = states.copy()
    actions = actions.copy()
    if gripper_input_range is not None:
        low, high = gripper_input_range
        states[..., [6, 13]] = _normalize_range(states[..., [6, 13]], low,
                                                high)
        actions[..., [6, 13]] = _normalize_range(actions[..., [6, 13]], low,
                                                 high)
    states[..., [6, 13]] = _aloha_gripper_to_angular(states[..., [6, 13]])
    actions[..., [6, 13]] = _aloha_gripper_from_angular_inv(actions[...,
                                                                    [6, 13]])
    return states, actions


def _build_action_chunks(actions: np.ndarray, states: np.ndarray, horizon: int,
                         window_start_idx: int, delta_mask: tuple[bool, ...],
                         repeat_terminal: bool) -> np.ndarray:
    num_frames, action_dim = actions.shape
    if num_frames == 0:
        return np.empty((0, action_dim), dtype=np.float32)
    offsets = window_start_idx + np.arange(horizon, dtype=np.int64)
    target_indices = np.arange(num_frames, dtype=np.int64)[:, None] + offsets
    if repeat_terminal:
        target_indices = np.minimum(target_indices, num_frames - 1)
        chunks = actions[target_indices]
    else:
        valid = target_indices < num_frames
        chunks = actions[np.minimum(target_indices, num_frames - 1)][valid]
        chunks = chunks.reshape(-1, action_dim)

    if repeat_terminal:
        chunks = chunks.reshape(-1, action_dim)

    if delta_mask:
        if len(delta_mask) > action_dim or len(delta_mask) > states.shape[-1]:
            raise ValueError(
                f'Delta mask length {len(delta_mask)} exceeds state/action '
                f'dimensions {states.shape[-1]}/{action_dim}.')
        mask = np.asarray(delta_mask, dtype=bool)
        if repeat_terminal:
            current = np.repeat(states[:, None, :len(mask)], horizon, axis=1)
            current = current.reshape(-1, len(mask))
        else:
            target_indices = np.arange(num_frames)[:, None] + offsets
            valid = target_indices < num_frames
            current = np.broadcast_to(states[:, None, :len(mask)],
                                      (num_frames, horizon, len(mask)))[valid]
        chunks[:, :len(mask)] -= np.where(mask, current, 0.0)
    return chunks.astype(np.float32, copy=False)


def _stats(values: np.ndarray) -> dict[str, list[float] | int]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            f'Cannot compute statistics for shape {values.shape}.')
    # Compute one dimension at a time so quantile does not materialize a full
    # copy of large action memmaps.
    q01 = []
    q99 = []
    for dim in range(values.shape[1]):
        column = np.asarray(values[:, dim])
        q01.append(float(np.quantile(column, 0.01)))
        q99.append(float(np.quantile(column, 0.99)))
    return {
        'mean':
        np.asarray(values.mean(axis=0, dtype=np.float64),
                   dtype=np.float64).tolist(),
        'std':
        np.asarray(values.std(axis=0, dtype=np.float64),
                   dtype=np.float64).tolist(),
        'min':
        np.asarray(values.min(axis=0), dtype=np.float64).tolist(),
        'max':
        np.asarray(values.max(axis=0), dtype=np.float64).tolist(),
        'q01':
        q01,
        'q99':
        q99,
        'count':
        int(values.shape[0]),
    }


def compute_statistics(dataset_paths: Sequence[Path],
                       profile_name: str = 'auto',
                       state_key: str | None = None,
                       action_key: str | None = None,
                       delta_mask: tuple[bool, ...] | None = None,
                       action_horizon: int = 50,
                       window_start_idx: int = 0,
                       repeat_terminal: bool = True,
                       signs: tuple[float, ...] | None = None,
                       aloha_gripper_input_range: tuple[float, float]
                       | None = None,
                       statistic_name: str = 'private',
                       temp_dir: Path | None = None) -> tuple[dict, dict]:
    """Compute transformed state/action statistics and provenance metadata."""
    if action_horizon <= 0:
        raise ValueError('action_horizon must be positive')
    if window_start_idx < 0:
        raise ValueError('window_start_idx must be non-negative')
    roots = _discover_dataset_roots(dataset_paths)
    files = _parquet_files(roots)

    requested = None if profile_name == 'auto' else PROFILES[profile_name]
    initial_state_key = state_key or (requested.state_key
                                      if requested else 'observation.state')
    initial_action_key = action_key or (requested.action_key
                                        if requested else 'action')
    first_states, first_actions = next(
        _iter_episodes(files, initial_state_key, initial_action_key))
    if requested is None:
        if delta_mask is not None:
            requested = Profile(initial_state_key, initial_action_key,
                                delta_mask)
        else:
            requested = _infer_profile(first_states.shape[-1],
                                       initial_state_key, initial_action_key)
    state_key = state_key or requested.state_key
    action_key = action_key or requested.action_key
    delta_mask = requested.delta_mask if delta_mask is None else delta_mask
    signs = requested.signs if signs is None else signs
    if aloha_gripper_input_range is None:
        aloha_gripper_input_range = requested.aloha_gripper_input_range
    use_aloha_coordinates = profile_name == 'aloha'

    if state_key != initial_state_key or action_key != initial_action_key:
        first_states, first_actions = next(
            _iter_episodes(files, state_key, action_key))
    state_dim = first_states.shape[-1]
    action_dim = first_actions.shape[-1]
    if state_dim != action_dim and delta_mask:
        raise ValueError(
            'Delta conversion requires compatible state/action dimensions; '
            f'got {state_dim} and {action_dim}.')

    total_states = 0
    total_actions = 0
    num_episodes = 0
    for states, actions in _iter_episodes(files, state_key, action_key):
        if states.shape[-1] != state_dim or actions.shape[-1] != action_dim:
            raise ValueError('State/action dimensions vary across episodes.')
        total_states += len(states)
        if repeat_terminal:
            total_actions += len(states) * action_horizon
        else:
            total_actions += sum(
                max(
                    0,
                    min(action_horizon,
                        len(states) - index - window_start_idx))
                for index in range(len(states)))
        num_episodes += 1

    if temp_dir is not None:
        temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
            dir=None if temp_dir is None else str(temp_dir),
            prefix='transformed-stats-') as work_dir:
        work_dir = Path(work_dir)
        state_values = np.memmap(
            work_dir / 'states.bin',
            mode='w+',
            dtype=np.float32,
            shape=(total_states, state_dim))
        action_values = np.memmap(
            work_dir / 'actions.bin',
            mode='w+',
            dtype=np.float32,
            shape=(total_actions, action_dim))
        state_offset = 0
        action_offset = 0
        for episode_number, (states, actions) in enumerate(
                _iter_episodes(files, state_key, action_key), start=1):
            states = _apply_signs(states.astype(np.float32, copy=False), signs)
            actions = _apply_signs(
                actions.astype(np.float32, copy=False), signs)
            if use_aloha_coordinates:
                states, actions = _apply_aloha_coordinates(
                    states, actions, aloha_gripper_input_range)
            chunks = _build_action_chunks(actions, states, action_horizon,
                                          window_start_idx, delta_mask,
                                          repeat_terminal)
            state_values[state_offset:state_offset + len(states)] = states
            action_values[action_offset:action_offset + len(chunks)] = chunks
            state_offset += len(states)
            action_offset += len(chunks)
            if episode_number % 25 == 0 or episode_number == num_episodes:
                print(
                    f'Processed {episode_number}/{num_episodes} episodes',
                    file=sys.stderr,
                    flush=True)
        state_values.flush()
        action_values.flush()
        result = {
            statistic_name: {
                'proprio': _stats(state_values),
                'action': _stats(action_values),
            }
        }
        # Close the mmap handles before TemporaryDirectory removes the files.
        # Local filesystems permit unlinking open files, but NFS/CPFS may keep
        # them as .nfs* entries and make cleanup fail with "Directory not
        # empty".
        del state_values
        del action_values

    metadata = {
        'dataset_roots': [str(root) for root in roots],
        'profile': profile_name,
        'state_key': state_key,
        'action_key': action_key,
        'state_dim': state_dim,
        'action_dim': action_dim,
        'delta_mask': list(delta_mask),
        'signs': None if signs is None else list(signs),
        'aloha_gripper_input_range': aloha_gripper_input_range,
        'action_horizon': action_horizon,
        'window_start_idx': window_start_idx,
        'repeat_terminal': repeat_terminal,
        'episodes': num_episodes,
        'state_samples': total_states,
        'action_samples': total_actions,
    }
    return result, metadata


def _config_value(config: Any, key: str, default=None):
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _source_configs(dataset_config: Any) -> list[Any]:
    """Return non-grouped ParquetDataset configs from a wrapper config."""
    datasets = _config_value(dataset_config, 'datasets')
    if isinstance(datasets, Mapping) and _config_value(datasets, 'type'):
        return [datasets]
    if isinstance(datasets, (list, tuple)):
        return list(datasets)
    raise ValueError(
        'Automatic transformed statistics currently requires a single/list '
        'ParquetDataset configuration. Grouped datasets should configure '
        'statistics per group explicitly.')


def _common_source_value(source_configs: Sequence[Any], key: str, default):
    values = [_config_value(config, key, default) for config in source_configs]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ValueError(
            f'Automatic transformed statistics requires a common {key!r}; '
            'got '
            f'{values}.')
    return first


def compute_statistics_from_dataset_config(
        dataset_config: Any,
        options: Mapping[str, Any] | None = None,
        default_temp_dir: Path | None = None) -> tuple[dict, dict]:
    """Compute transformed statistics from a training dataset configuration.

    Dataset roots, action horizon, terminal-padding behavior, and statistic
    name are inherited from the actual training config. ``options`` only
    describes preprocessing that cannot be recovered safely from generic
    dataset fields, such as the robot profile or an explicit delta mask.
    """
    options = dict(options or {})
    profile_name = options.pop('profile', options.pop('profile_name', 'auto'))
    source_configs = _source_configs(dataset_config)

    dataset_paths = []
    for source_config in source_configs:
        source_type = _config_value(source_config, 'type')
        if source_type != 'ParquetDataset':
            raise ValueError('Automatic transformed statistics only supports '
                             'ParquetDataset '
                             f'sources, got {source_type!r}.')
        roots = _config_value(source_config, 'data_root_path')
        if isinstance(roots, (str, Path)):
            roots = [roots]
        if not roots:
            raise ValueError(
                'ParquetDataset.data_root_path must not be empty.')
        dataset_paths.extend(Path(root) for root in roots)

    defaults = {
        'action_horizon':
        _common_source_value(source_configs, 'action_window_size', 9),
        'window_start_idx':
        _common_source_value(source_configs, 'window_start_idx', 1),
        'repeat_terminal':
        _common_source_value(source_configs, 'supervise_terminal_padding',
                             False),
        'statistic_name':
        _config_value(dataset_config, 'statistic_name', 'private'),
        'temp_dir':
        default_temp_dir,
    }
    for key, value in defaults.items():
        options.setdefault(key, value)

    supported = {
        'state_key', 'action_key', 'delta_mask', 'action_horizon',
        'window_start_idx', 'repeat_terminal', 'signs',
        'aloha_gripper_input_range', 'statistic_name', 'temp_dir'
    }
    unknown = sorted(set(options) - supported)
    if unknown:
        raise ValueError(
            f'Unknown automatic transformed statistics options: {unknown}')

    for key in ('delta_mask', 'signs'):
        if options.get(key) is not None:
            options[key] = tuple(options[key])
    if options.get('aloha_gripper_input_range') is not None:
        options['aloha_gripper_input_range'] = tuple(
            options['aloha_gripper_input_range'])
    if options.get('temp_dir') is not None:
        options['temp_dir'] = Path(options['temp_dir'])

    return compute_statistics(
        dataset_paths=dataset_paths, profile_name=profile_name, **options)


def _parse_signs(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    signs = tuple(float(item) for item in value.split(','))
    if not signs or any(sign not in {-1.0, 1.0} for sign in signs):
        raise argparse.ArgumentTypeError('signs must contain only -1 or 1')
    return signs


def _parse_range(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = tuple(float(item) for item in value.split(','))
    if len(parts) != 2 or parts[1] <= parts[0]:
        raise argparse.ArgumentTypeError(
            'range must be an increasing LOW,HIGH pair')
    return parts


def _format_python(stats: dict, variable_name: str) -> str:
    body = pprint.pformat(stats, width=79, sort_dicts=False)
    return f'{variable_name} = {body}\n'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'data_paths',
        nargs='+',
        type=Path,
        help='LeRobot dataset root(s), or parent directories containing them.')
    parser.add_argument(
        '--profile',
        choices=('auto', *PROFILES),
        default='auto',
        help='Robot action-space preset. Default: infer from keys/dimensions.')
    parser.add_argument('--state-key', help='Override the parquet state key.')
    parser.add_argument(
        '--action-key', help='Override the parquet action key.')
    parser.add_argument(
        '--delta-mask',
        type=_parse_bool_mask,
        help='Comma-separated mask; true dimensions become action - state.')
    parser.add_argument(
        '--no-delta',
        action='store_true',
        help='Keep every action dimension absolute.')
    parser.add_argument('--signs', type=_parse_signs)
    parser.add_argument(
        '--gripper-input-range',
        type=_parse_range,
        help='ALOHA raw gripper LOW,HIGH range. The aloha profile defaults '
        'to -0.01,0.08 to match the checked-in configs.')
    parser.add_argument('--action-horizon', type=int, default=50)
    parser.add_argument(
        '--window-start-index',
        type=int,
        default=0,
        help='First action offset used to build action chunks. Default: 0.')
    parser.add_argument(
        '--exclude-terminal-padding',
        action='store_true',
        help='Exclude out-of-episode future actions instead of repeating the '
        'last action.')
    parser.add_argument('--temp-dir', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--format', choices=('python', 'json'), default='python')
    parser.add_argument(
        '--statistic-name',
        default='private',
        help='Top-level statistics key used by the target config.')
    parser.add_argument(
        '--variable-name', default='_TRANSFORMED_DATASET_STATS')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_delta and args.delta_mask is not None:
        raise ValueError('--no-delta and --delta-mask are mutually exclusive')
    delta_mask = () if args.no_delta else args.delta_mask
    stats, metadata = compute_statistics(
        args.data_paths,
        profile_name=args.profile,
        state_key=args.state_key,
        action_key=args.action_key,
        delta_mask=delta_mask,
        action_horizon=args.action_horizon,
        window_start_idx=args.window_start_index,
        repeat_terminal=not args.exclude_terminal_padding,
        signs=args.signs,
        aloha_gripper_input_range=args.gripper_input_range,
        statistic_name=args.statistic_name,
        temp_dir=args.temp_dir)

    if args.format == 'python':
        output = _format_python(stats, args.variable_name)
    else:
        output = json.dumps({
            'norm_stats': stats,
            'metadata': metadata
        },
                            indent=2) + '\n'
    if args.output is None:
        print(output, end='')
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding='utf-8')
        print(f'Wrote statistics to {args.output}')
    print(json.dumps(metadata, indent=2), file=sys.stderr)


if __name__ == '__main__':
    main()
