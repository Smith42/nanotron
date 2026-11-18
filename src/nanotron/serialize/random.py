from pathlib import Path

import numpy as np
import torch
from numpy.core.multiarray import _reconstruct

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

    # TODO @thomasw21: That's annothing but this actually uses pickle, we might need to change that for something else
    torch.save(random_states, filename)


def load_random_states(parallel_context: ParallelContext, root_folder: Path):
    # TODO @thomasw21: This basically assumes that we have exactly the same topology as the one we used when saving.
    filename = (
        root_folder
        / "random"
        / f"tp-{dist.get_rank(parallel_context.tp_pg)}-of-{parallel_context.tp_pg.size()}_dp-{dist.get_rank(parallel_context.dp_pg)}-of-{parallel_context.dp_pg.size()}_pp-{dist.get_rank(parallel_context.pp_pg)}-of-{parallel_context.pp_pg.size()}.pt"
    )

    with torch.serialization.safe_globals([
        RandomState, RandomStates, np.ndarray, np.dtype, type(np.dtype(np.uint32)), _reconstruct,
    ]):
        return torch.load(filename, map_location="cpu", weights_only=True)
