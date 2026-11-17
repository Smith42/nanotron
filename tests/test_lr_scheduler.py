"""CPU regression coverage for checkpoint scheduler restoration."""

from copy import deepcopy

import pytest
import torch

from nanotron.config import LRSchedulerArgs
from nanotron.optim.named_optimizer import NamedOptimizer
from nanotron.serialize import optimizer as serialization


@pytest.mark.parametrize("saved_step", [2, 6])
@pytest.mark.parametrize("scheduler_kind", ["torch", "nanotron"])
def test_resume_preserves_scheduler_and_learning_rates(tmp_path, monkeypatch, saved_step, scheduler_kind):
    if scheduler_kind == "nanotron":
        if not torch.cuda.is_available():
            pytest.skip("Nanotron builder imports FlashAttention, which initializes CUDA")
        from nanotron.helpers import lr_scheduler_builder

        monkeypatch.setattr("nanotron.helpers.log_rank", lambda *args, **kwargs: None)
    else:
        def lr_scheduler_builder(optimizer, args, total_training_steps):
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer.get_base_optimizer(), lambda step: min(step / 4, 1)
            )

    def make_optimizer():
        groups = []
        for name, lr in [("a", 0.001), ("b", 0.002)]:
            param = torch.nn.Parameter(torch.zeros(1))
            param.grad = torch.ones_like(param)
            groups.append({"named_params": [(name, param)], "lr": lr})
        return NamedOptimizer(groups, torch.optim.AdamW)

    args = LRSchedulerArgs(
        learning_rate=0.001,
        lr_warmup_steps=4,
        lr_warmup_style="linear",
        lr_decay_style="cosine",
        min_decay_lr=0.00001,
    )
    optimizer = make_optimizer()
    scheduler = lr_scheduler_builder(optimizer, args, total_training_steps=10)
    for _ in range(saved_step):
        optimizer.step()
        scheduler.step()

    checkpoint = tmp_path / "lr_scheduler"
    checkpoint.mkdir()
    torch.save(scheduler.state_dict(), checkpoint / "scheduler.pt")
    monkeypatch.setattr(serialization, "lr_scheduler_filename", lambda *args: "scheduler.pt")

    # Match the trainer: load the optimizer, construct the scheduler, then restore it.
    resumed_optimizer = make_optimizer()
    resumed_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    resumed = lr_scheduler_builder(resumed_optimizer, args, total_training_steps=10)
    serialization.load_lr_scheduler(resumed, False, None, tmp_path)

    assert resumed.state_dict() == scheduler.state_dict()
    for step in range(saved_step, 11):
        assert resumed.last_epoch == scheduler.last_epoch == step
        assert resumed.get_last_lr() == scheduler.get_last_lr()
        assert [group["lr"] for group in resumed_optimizer.param_groups] == scheduler.get_last_lr()
        if step < 10:
            optimizer.step()
            scheduler.step()
            resumed_optimizer.step()
            resumed.step()
