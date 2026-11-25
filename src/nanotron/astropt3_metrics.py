"""AstroPT step diagnostics, independent of the family-balanced training objective."""

import torch
import torch.distributed as dist

FAMILIES = ("image", "spectrum", "scalar")


def reduce_loss_metrics(outputs, modality_names, dp_pg=None):
    """Pool (loss sum, scored-target count) rows across microbatches and DP peers only."""
    totals = torch.stack([output["loss_stats"] for output in outputs]).sum(dim=0)
    if dp_pg is not None and dp_pg.size() > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=dp_pg)

    names = [*modality_names, *(f"{family}_family" for family in FAMILIES)]
    metrics = {}
    for name, (loss_sum, count) in zip(names, totals.tolist(), strict=True):
        metrics[f"{name}_loss_count"] = int(round(count)) if count == 0 or count >= 0.5 else 1
        if count:
            metrics[f"{name}_loss"] = loss_sum / count
    return metrics
