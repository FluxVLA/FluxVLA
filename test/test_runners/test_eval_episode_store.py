# Copyright 2026 Limx Dynamics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock, sentinel

import accelerate
import pytest

from fluxvla.engines.utils.eval_episode_store import EvalEpisodeStore
from fluxvla.engines.utils.overwatch import DistributedOverwatch


@pytest.fixture
def store(tmp_path):
    directory = tmp_path / 'EVAL-libero_10-gpt6-astra-test'
    directory.mkdir()
    return EvalEpisodeStore(
        directory,
        dict(
            task_suite_name='libero_10',
            model_family='gpt6-astra',
            task_ids=[0, 1],
            num_trials_per_task=3,
            seed=7))


def _record(task=0, trial=0, success=False):
    return dict(
        task_suite='libero_10',
        task_id=task,
        trial_id=trial,
        status='completed',
        success=success,
        duration_seconds=12.5,
        start_time=100)


def test_recover_both_success_and_failure_but_not_interrupted_trial(store):
    (store.run_dir / 'rank0.txt').write_text(
        'Evaluating Task 0, Trial 0\nSuccess: True\n'
        'Evaluating Task 0, Trial 2\nStarting episode 3...\n')
    (store.run_dir /
     'rank1.txt').write_text('Evaluating Task 0, Trial 1\nSuccess: False\n')
    records = store.prepare(resume=True)
    assert set(records) == {(0, 0), (0, 1)}
    assert records[(0, 0)]['success'] is True
    assert records[(0, 1)]['success'] is False
    assert records[(0, 0)]['duration_seconds'] is None
    assert len(list(store.episode_dir.glob('*.json'))) == 2


def test_stored_result_survives_without_final_log_or_summary(store):
    assert store.prepare() == {}
    record = _record(success=True)
    store.save(record)
    restored = store.prepare(resume=True)
    assert restored == {(0, 0): record}
    assert not (store.run_dir / 'summary.json').exists()


def test_resume_is_idempotent_and_keeps_recorded_timings(store):
    store.prepare()
    record = _record()
    store.save(record)
    (store.run_dir /
     'rank0.txt').write_text('Evaluating Task 0, Trial 0\nSuccess: False\n')
    assert store.prepare(resume=True) == {(0, 0): record}
    assert store.prepare(resume=True) == {(0, 0): record}


def test_resume_rejects_changed_trial_count_or_seed(store):
    store.prepare()
    for change in ({'num_trials_per_task': 5}, {'seed': 8}):
        other = EvalEpisodeStore(store.run_dir, {**store.metadata, **change})
        with pytest.raises(ValueError, match='settings differ'):
            other.prepare(resume=True)


def test_resume_rejects_conflicting_legacy_results(store):
    for rank, success in enumerate((True, False)):
        (store.run_dir / f'rank{rank}.txt'
         ).write_text(f'Evaluating Task 0, Trial 0\nSuccess: {success}\n')
    with pytest.raises(ValueError, match='Conflicting legacy outcome'):
        store.prepare(resume=True)


def test_resume_rejects_bad_trial_ids_and_incomplete_records(store):
    for record in (_record(trial=5), {**_record(), 'status': 'incomplete'}):
        with pytest.raises(ValueError, match='Invalid'):
            store.save(record)


def test_resume_new_run_does_not_overwrite_existing_progress(store):
    store.prepare()
    store.save(_record())
    with pytest.raises(ValueError, match='already contains episodes'):
        store.prepare()


def test_partial_temporary_files_are_ignored(store):
    store.prepare()
    store.save(_record())
    (store.episode_dir / '.task0_trial1.json.partial').write_text('{')
    assert len(store.prepare(resume=True)) == 1
    saved = json.loads((store.episode_dir / 'task0_trial0.json').read_text())
    assert saved['success'] is False


@pytest.mark.parametrize('rank,local_rank', [(0, 0), (3, 1)])
def test_distributed_overwatch_uses_process_state(monkeypatch, caplog, rank,
                                                  local_rank):
    state = SimpleNamespace(
        is_main_process=rank == 0,
        process_index=rank,
        local_process_index=local_rank,
        num_processes=4)
    state_factory = Mock(return_value=state)
    monkeypatch.setattr(accelerate, 'PartialState', state_factory)
    caplog.set_level(logging.WARNING, logger='test_overwatch')
    overwatch = DistributedOverwatch('test_overwatch')
    state_factory.assert_called_once_with()
    assert overwatch.distributed_state is state
    assert overwatch.is_rank_zero() == (rank == 0)
    assert overwatch.rank() == rank
    assert overwatch.local_rank() == local_rank
    assert overwatch.world_size() == 4
    assert overwatch.logger.getEffectiveLevel() == (
        logging.INFO if rank == 0 else logging.ERROR)


def test_distributed_overwatch_forwards_rank_guards(monkeypatch, caplog):
    state = SimpleNamespace(
        is_main_process=True,
        on_main_process=sentinel.on_main,
        on_local_main_process=sentinel.on_local,
        main_process_first=sentinel.main_first,
        local_main_process_first=sentinel.local_first)
    monkeypatch.setattr(accelerate, 'PartialState', Mock(return_value=state))
    caplog.set_level(logging.WARNING, logger='test_overwatch_guards')
    overwatch = DistributedOverwatch('test_overwatch_guards')
    assert overwatch.rank_zero_only is state.on_main_process
    assert overwatch.local_zero_only is state.on_local_main_process
    assert overwatch.rank_zero_first is state.main_process_first
    assert overwatch.local_zero_first is state.local_main_process_first
