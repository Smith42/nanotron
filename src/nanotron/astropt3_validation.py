"""Startup distribution and evaluation of the fixed AstroPT3 validation panel."""

import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from nanotron.astropt3_metrics import FAMILIES, reduce_loss_metrics


def load_validation_shard(config, model_config, context, checkpoint):
    def load():
        from astropt3.data.validation_panel import PANEL_FILENAME, load_validation_panel, save_validation_manifest

        ids, shards = load_validation_panel(
            model_config,
            config.tokens.validation_objects,
            config.tokens.sequence_length,
            config.tokens.micro_batch_size,
            context.dp_pg.size(),
        )
        path = Path(config.checkpoints.checkpoints_path) / PANEL_FILENAME
        previous = Path(checkpoint).parent / PANEL_FILENAME if checkpoint else None
        reference = config.tokens.validation_manifest
        if reference is not None:
            previous = Path(reference)
        elif previous is not None and not previous.is_file():
            previous = None
        save_validation_manifest(path, ids, reference=previous)
        return shards

    if context.world_pg.size() == 1:
        return load()[0]

    # Fetch once, scatter CPU batches by DP replica, and give every TP peer the exact same tensors.
    group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
    assert isinstance(group, dist.ProcessGroup)
    payload = None
    error = None
    try:
        if dist.get_rank(group) == 0:
            try:
                shards = load()
                payload = [shards[context.get_local_ranks(rank)["dp"]] for rank in range(group.size())]
            except Exception as exc:
                error = exc
                payload = [f"{type(exc).__name__}: {exc}"] * group.size()
        local = [None]
        dist.scatter_object_list(local, payload, src=0, group=group)
        if isinstance(local[0], str):
            raise RuntimeError(f"Could not load AstroPT3 validation panel: {local[0]}") from error
        return local[0]
    finally:
        dist.destroy_process_group(group)


@torch.no_grad()
def evaluate_validation_shard(model, batches, modality_names, device, dp_pg):
    """Score the DP shard without touching DDP, gradients or training-time noise."""
    started = time.perf_counter()
    names = tuple(modality_names)
    was_training = model.training
    model.eval()
    try:
        # PP=1 needs no pipeline engine, and the DDP wrapper could synchronize uneven shards.
        outputs = [model(**{k: v.to(device) for k, v in batch.items()}) for batch in batches]
        if not outputs:
            empty = torch.zeros((len(names) + len(FAMILIES), 2), dtype=torch.float64, device=device)
            outputs = [{"loss_stats": empty}]
        metrics = reduce_loss_metrics(outputs, names, dp_pg, loss_prefix="val/", count_prefix="loss_count/val/")
    finally:
        model.train(was_training)
    return {**metrics, "val/seconds": time.perf_counter() - started}
