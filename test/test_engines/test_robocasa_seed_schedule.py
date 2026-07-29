from fluxvla.engines.runners.robocasa_eval_runner import RobocasaEvalRunner


def test_episode_seed_id_uses_fixed_protocol_stride():
    runner = object.__new__(RobocasaEvalRunner)
    runner.episode_seed_stride = 50

    assert runner._episode_seed_id(task_id=7, trial_id=2) == 352
