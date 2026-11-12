import json

from fluxvla.engines.runners.libero_eval_runner import LiberoEvalRunner


def test_write_action_trace(tmp_path):
    runner = LiberoEvalRunner.__new__(LiberoEvalRunner)
    runner.save_action_traces = True
    runner.result_output_dir = str(tmp_path)
    runner.run_dir = str(tmp_path / "unused")
    runner.task_suite_name = "libero_spatial"
    rows = [{"step": 3, "denormalized_action": [0.1] * 7, "done": False}]

    runner._write_action_trace(rows, task_id=2, trial_id=4)

    trace_path = tmp_path / "libero_spatial" / "traces" / "task2_trial4.jsonl"
    assert json.loads(trace_path.read_text(encoding="utf-8")) == rows[0]


def test_write_action_trace_can_be_disabled(tmp_path):
    runner = LiberoEvalRunner.__new__(LiberoEvalRunner)
    runner.save_action_traces = False
    runner.result_output_dir = str(tmp_path)
    runner.run_dir = str(tmp_path / "unused")
    runner.task_suite_name = "libero_spatial"

    runner._write_action_trace([], task_id=0, trial_id=0)

    assert not (tmp_path / "libero_spatial" / "traces").exists()
