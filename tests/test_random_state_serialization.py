from types import SimpleNamespace

import torch

from nanotron import distributed as dist
from nanotron.random import RandomStates, get_current_random_state
from nanotron.serialize.random import load_random_states, save_random_states


def test_rng_checkpoint_loads_without_custom_pickle_globals(tmp_path, monkeypatch):
    pg = SimpleNamespace(size=lambda: 1)
    context = SimpleNamespace(tp_pg=pg, dp_pg=pg, pp_pg=pg)
    monkeypatch.setattr(dist, "get_rank", lambda group: 0)
    states = RandomStates({"tp_synced": get_current_random_state()})

    save_random_states(states, context, tmp_path)
    (checkpoint,) = (tmp_path / "random").glob("*.pt")
    assert set(torch.load(checkpoint, weights_only=True, map_location="cpu")) == {"tp_synced"}
    monkeypatch.delattr(torch.serialization, "safe_globals", raising=False)
    assert load_random_states(context, tmp_path) == states
