from types import SimpleNamespace

import pytest
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from fluxvla.mcore.pi05.scheduler import apply_initial_lr_contract


class _Scheduler:
    def __init__(self):
        self.num_steps = 0
        self.optimizer = SimpleNamespace(param_groups=[{"lr": 5e-5}, {"lr": 1e-5}])


def test_fluxvla_initial_lr_contract_only_zeros_current_lr():
    scheduler = _Scheduler()

    apply_initial_lr_contract(scheduler, "fluxvla")

    assert scheduler.num_steps == 0
    assert [group["lr"] for group in scheduler.optimizer.param_groups] == [0.0, 0.0]


def test_fluxvla_contract_matches_two_step_huggingface_cosine_lr_sequence():
    optimizer = SimpleNamespace(param_groups=[{"lr": 5e-5}])
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=5e-5,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=2,
        lr_decay_style="cosine",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=2,
        wd_incr_style="constant",
    )

    apply_initial_lr_contract(scheduler, "fluxvla")
    update_lrs = [optimizer.param_groups[0]["lr"]]
    scheduler.step(1)
    update_lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step(1)
    post_training_lr = optimizer.param_groups[0]["lr"]

    assert update_lrs == pytest.approx([0.0, 2.5e-5])
    assert post_training_lr == pytest.approx(0.0)


def test_megatron_initial_lr_contract_is_unchanged():
    scheduler = _Scheduler()

    apply_initial_lr_contract(scheduler, "megatron")

    assert [group["lr"] for group in scheduler.optimizer.param_groups] == [5e-5, 1e-5]


def test_fluxvla_initial_lr_contract_rejects_resume_state():
    scheduler = _Scheduler()
    scheduler.num_steps = 1

    with pytest.raises(RuntimeError, match="fresh scheduler"):
        apply_initial_lr_contract(scheduler, "fluxvla")
