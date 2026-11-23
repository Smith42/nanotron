from pathlib import Path

import numpy as np
import torch

from nanotron import distributed as dist
from nanotron.parallel import ParallelContext
from nanotron.random import RandomState, RandomStates


def save_random_states(
    random_states: RandomStates,
    parallel_context: ParallelContext,
    root_folder: Path,
):
    """All processes save their own random state"""
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}.pt"
    )
    filename.parent.mkdir(exist_ok=True, parents=True)

    payload = {}
    for key, state in random_states.items():
        algorithm, keys, pos, has_gauss, cached = state.numpy
        payload[key] = (
            state.random,
            (algorithm, keys.tolist(), pos, has_gauss, cached),
            state.torch_cpu,
            state.torch_cuda,
        )
    torch.save(payload, filename)


def load_random_states(parallel_context: ParallelContext, root_folder: Path):
    # TODO @thomasw21: This basically assumes that we have exactly the same topology as the one we used when saving.
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}.pt"
    )

    payload = torch.load(filename, map_location="cpu", weights_only=True)
    states = {}
    for key, (python_state, numpy_state, torch_cpu, torch_cuda) in payload.items():
        algorithm, keys, pos, has_gauss, cached = numpy_state
        states[key] = RandomState(
            python_state, (algorithm, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached), torch_cpu, torch_cuda
        )
    return RandomStates(states)
