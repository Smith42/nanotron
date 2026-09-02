"""AstroPT3 training model: SmolLM3 (qwen2-style) decoder body with
continuous-modality inputs and per-modality regression heads.

This mirrors the HF release implementation in the ``astropt3`` package
(``astro/src/astropt3/modeling_astropt3.py``), replacing exactly two blocks of
the upstream Qwen2 pipeline graph:

- the vocab ``TensorParallelEmbedding`` block becomes :class:`AstroPT3Embedding`:
  a 64-id special-token embedding plus additive per-modality deltas
  ``encoder_m(value) + pos_embed_m(position)`` at placeholder slots, plus the
  per-modality flow (data -> latent z, with logdet) jetformer needs;
- the ``lm_head`` + sharded-CE ``Loss`` blocks become :class:`AstroPT3ModalityHead`
  (per-modality GMM heads applied one position LEFT of each modality token —
  astroPT's ``starts-1`` alignment) plus :class:`AstroPT3Loss` (the family-
  balanced mean of per-modality ``NLL_GMM(z) - logdet`` losses).

Parallelism contract (see astro/PLAN.md):

- **PP=1 always** (asserted): the whole micro-batch dict reaches every rank,
  so dict-valued inputs pass through PipelineBlocks locally and modality
  tensors never cross pipeline stages.
- **TP**: the transformer body is sharded as upstream. Modality
  encoders/decoders/pos-embedders are tiny layers kept **replicated**
  across TP ranks: they are plain ``nn.Linear``/``nn.Embedding`` modules, so
  ``mark_unsharded_params_as_tied_across_tp`` ties them across the TP group
  (grads identical by design under ALL_REDUCE because every TP rank sees the
  same inputs and the same replicated hidden states). ``tp_mode`` is asserted
  to be ALL_REDUCE: REDUCE_SCATTER shards the hidden stream over the sequence
  which breaks the replication argument (revisit if throughput demands).
- Sequence packing: ``position_ids`` restart at 0 per object and pads sit at
  position 0, so the ``cu_seqlens`` derived from zeros gives each object (and
  each pad token) its own attention segment — same doc mask the HF side gets
  from transformers' ``create_causal_mask``.

Batch contract (built by ``astropt3.data.nanotron_loader``): flat dict of
``input_ids`` [b,s], ``position_ids`` [b,s], and per modality ``{m}_values``
[n_m, input_size], ``{m}_positions`` (long [n_m] or float [n_m, pos_dim]) and
``{m}_mask`` bool [b,s], flattened in row-major (batch, time) order. A
modality absent from a micro-batch ships zero-length tensors; its modules
still participate in autograd (with zero gradient) so DDP never sees unused
parameters.
"""

import math
import os
from contextlib import nullcontext
from typing import Dict, Optional, Union, cast

import torch
from torch import nn
from torch.nn import functional as F

from nanotron import distributed as dist
from nanotron import logging
from nanotron.config import Config, ParallelismArgs
from nanotron.config.astropt3_config import AstroPT3Config
from nanotron.config.models_config import RandomInit, SpectralMupInit
from nanotron.logging import LoggingCollectorMixin, log_rank
from nanotron.models import NanotronModel
from nanotron.models.qwen import Qwen2DecoderLayer, get_flops
from nanotron.nn.layer_norm import LlamaRMSNorm as RMSNorm
from nanotron.nn.layer_norm import TritonRMSNorm
from nanotron.parallel import ParallelContext
from nanotron.parallel.parameters import NanotronParameter
from nanotron.parallel.pipeline_parallel.block import PipelineBlock, TensorPointer
from nanotron.parallel.pipeline_parallel.p2p import P2P
from nanotron.parallel.tensor_parallel.nn import (
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    TensorParallelLinearMode,
)
from nanotron.random import RandomStates, branch_random_state
from nanotron.scaling.parametrization import SpectralMupParametrizator, StandardParametrizator

logger = logging.get_logger(__name__)


# --- modality modules -------------------------------------------------------
# Deliberately duplicated from the HF-side astropt3.modalities (two
# implementations, one weight source of truth). Attribute names (c_fc, embed)
# are part of the conversion contract in tools/astropt3/convert_weights.py —
# keep them in sync.


class Encoder(nn.Module):
    """Data space -> embedding space: a single linear projection. Replicated across TP."""

    def __init__(self, hidden_size: int, in_size: int, bias: bool = False):
        super().__init__()
        self.c_fc = nn.Linear(in_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.c_fc(x)


class PositionEmbedder(nn.Module):
    """Per-modality positional embedding added at the input. Replicated across TP."""

    def __init__(self, hidden_size: int, modality: dict, bias: bool = False):
        super().__init__()
        self.pos_type = modality.get("pos_type", "index")
        if self.pos_type == "index":
            self.embed = nn.Embedding(modality.get("max_positions", 1024), hidden_size)
        elif self.pos_type == "continuous":
            self.embed = nn.Linear(modality.get("pos_input_size", 1), hidden_size, bias=bias)
        else:
            raise ValueError(f"unknown pos_type {self.pos_type!r}")

    def forward(self, pos):
        if self.pos_type == "index":
            return self.embed(pos)
        return self.embed(pos.to(self.embed.weight.dtype))


# --- jetformer regression-head modules --------------------------------------
# Duplicated from the HF-side astropt3.modalities with IDENTICAL attribute
# names (blocks.{i}.net.{0,2}, proj) — they are part of the conversion
# contract in tools/astropt3/convert_weights.py. Per-modality loss becomes
# mean(NLL_GMM(z) - logdet): exact likelihood in standardized patch space
# (may be negative).


class CouplingMLP(nn.Module):
    """RealNVP-style affine coupling over the feature dim of one token."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.split = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split)),
        )

    def forward(self, x, reverse: bool = False, flip: bool = False):
        x1 = x[..., : self.split]
        x2 = x[..., self.split :]
        ident, moved = (x2, x1) if flip else (x1, x2)
        s, t = self.net(ident).chunk(2, dim=-1)
        s = torch.tanh(s) * 1.5  # bound the scale for numerical stability
        if not reverse:
            moved = moved * torch.exp(s) + t
            logdet = s.sum(dim=-1)
        else:
            moved = (moved - t) * torch.exp(-s)
            logdet = -s.sum(dim=-1)
        halves = [moved, ident] if flip else [ident, moved]
        return torch.cat(halves, dim=-1), logdet


class TinyFlow1D(nn.Module):
    """Stack of affine couplings over (..., D) patch tokens. Replicated across TP."""

    def __init__(self, dim: int, steps: int = 4, hidden_dim: int = 128):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"TinyFlow1D requires an even token dim, got {dim}")
        self.blocks = nn.ModuleList(CouplingMLP(dim, hidden_dim) for _ in range(steps))

    def forward(self, x, reverse: bool = False):
        logdet = x.new_zeros(x.shape[:-1])
        indexed = list(enumerate(self.blocks))
        z = x
        for i, block in reversed(indexed) if reverse else indexed:
            z, ld = block(z, reverse=reverse, flip=(i % 2 == 1))
            logdet = logdet + ld
        return z, logdet


class GMMHead(nn.Module):
    """Embedding space -> raw GMM projection. Replicated across TP.

    Unlike the HF twin this returns the raw ``[n, K*(1+2D)]`` projection —
    a single tensor, so the pipeline-block key set keeps its shape — and the
    loss unpacks it via :func:`unpack_gmm_params`.
    """

    def __init__(self, hidden_size: int, out_size: int, k: int, bias: bool = False):
        super().__init__()
        self.k = k
        self.d = out_size
        self.proj = nn.Linear(hidden_size, k * (1 + 2 * out_size), bias=bias)

    def forward(self, h):
        return self.proj(h)


def unpack_gmm_params(pred: torch.Tensor, k: int, d: int):
    """[n, K*(1+2D)] raw projection -> (logits_pi, mu, log_sigma); mirrors the
    HF GMMHead.forward reshape and log-sigma clamp exactly."""
    out = pred.view(*pred.shape[:-1], k, 1 + 2 * d)
    logits_pi = out[..., 0]
    mu = out[..., 1 : 1 + d]
    log_sigma = out[..., 1 + d :].clamp(-7.0, 2.0)
    return logits_pi, mu, log_sigma


def gmm_nll(y, logits_pi, mu, log_sigma):
    """Per-token negative log-likelihood of y under the predicted GMM.

    y: (..., D); logits_pi: (..., K); mu/log_sigma: (..., K, D) -> (...,).
    """
    diff = y.unsqueeze(-2) - mu
    logp = (
        -0.5 * (diff.pow(2) * torch.exp(-2 * log_sigma)).sum(dim=-1)
        - log_sigma.sum(dim=-1)
        - 0.5 * mu.size(-1) * math.log(2 * math.pi)
    )
    return -torch.logsumexp(F.log_softmax(logits_pi, dim=-1) + logp, dim=-1)


def left_shift_mask(mask: torch.Tensor) -> torch.Tensor:
    """[b, s] bool -> True at t iff mask[t+1] is True (last column False).

    Hidden states at these positions predict the modality values at t+1
    (``<|begin_m|>`` predicts patch 0).
    """
    shifted = torch.zeros_like(mask)
    shifted[:, :-1] = mask[:, 1:]
    return shifted


# --- pipeline blocks --------------------------------------------------------


class AstroPT3Embedding(nn.Module):
    """64-id token embedding + additive modality deltas at placeholder slots."""

    def __init__(
        self,
        tp_pg: dist.ProcessGroup,
        config: AstroPT3Config,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.random_states = random_states
        tp_mode = (
            parallel_config.tp_mode
            if parallel_config is not None and parallel_config.tp_mode is not None
            else TensorParallelLinearMode.ALL_REDUCE
        )
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=tp_mode,
        )
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(config.hidden_size, config.modality(name)["input_size"])
                for name in config.modality_names()
            }
        )
        self.pos_embeds = nn.ModuleDict(
            {name: PositionEmbedder(config.hidden_size, config.modality(name)) for name in config.modality_names()}
        )
        # ADR 0008: scalar modalities never flow — raw normalized value is
        # both the embedded input and the GMM target (mirrors the HF side)
        self.scalar_names = {name for name in config.modality_names() if config.modality(name).get("scalar", False)}
        self.flows = nn.ModuleDict(
            {
                name: TinyFlow1D(
                    config.modality(name)["input_size"],
                    steps=config.jetformer_flow_steps,
                    hidden_dim=config.jetformer_flow_hidden,
                )
                for name in config.modality_names()
                if name not in self.scalar_names
            }
        )
        # Noise curriculum (HF twin: AstroPT3Model.set_jet_noise_frac):
        # sigma = noise_max + (noise_min - noise_max) * frac anneals
        # noise_max -> noise_min as frac goes 0 -> 1; the trainer drives
        # frac each step. Noise hits only the embedded z copy in training
        # mode — the emitted {m}_z target and {m}_logdet stay clean.
        self.jetformer_noise_max = config.jetformer_noise_max
        self.jetformer_noise_min = config.jetformer_noise_min
        self.jet_noise_frac = 1.0

    def set_jet_noise_frac(self, frac: float):
        self.jet_noise_frac = min(max(frac, 0.0), 1.0)

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch_size, seq_length]
        position_ids: torch.Tensor,  # [batch_size, seq_length]
        modality_values: Dict[str, torch.Tensor],  # name -> [n_m, input_size]
        modality_positions: Dict[str, torch.Tensor],  # name -> [n_m] or [n_m, pos_dim]
        modality_masks: Dict[str, torch.Tensor],  # name -> bool [batch_size, seq_length]
    ):
        input_embeds = self.token_embedding(input_ids.view(-1))  # [b*s, hidden]
        delta = torch.zeros_like(input_embeds)
        extras = {}
        # Always run every encoder (even on zero-length values): the empty
        # index_put keeps absent modalities in the autograd graph with zero
        # gradient, so DDP never sees unused parameters.
        for name, encoder in self.encoders.items():
            values = modality_values[name].to(input_embeds.dtype)
            if name not in self.scalar_names:
                z, logdet = self.flows[name](values)
                extras[f"{name}_z"] = z
                extras[f"{name}_logdet"] = logdet
                sigma = (
                    self.jetformer_noise_max
                    + (self.jetformer_noise_min - self.jetformer_noise_max) * self.jet_noise_frac
                )
                values = z
                if self.training and sigma > 0:
                    # the noise must be IDENTICAL across TP ranks or the
                    # replicated-hidden-stream contract breaks — draw it under
                    # the trainer's tp_synced random state
                    random_state = (
                        branch_random_state(self.random_states, "tp_synced", enabled=True)
                        if self.random_states is not None
                        and "tp_synced" in self.random_states
                        else nullcontext()
                    )
                    with random_state:
                        values = z + sigma * torch.randn_like(z)
            content = encoder(values) + self.pos_embeds[name](modality_positions[name]).to(input_embeds.dtype)
            delta = delta.index_put((modality_masks[name].view(-1),), content.to(input_embeds.dtype))
        return {"input_embeds": input_embeds + delta, "position_ids": position_ids, **extras}


class AstroPT3ModalityHead(nn.Module):
    """Per-modality regression decoders at ``starts-1``-aligned positions."""

    def __init__(self, config: AstroPT3Config):
        super().__init__()
        scalar_names = {
            name
            for name in config.modality_names()
            if config.modality(name).get("scalar", False)
        }
        decoders = {}
        for name in config.modality_names():
            input_size = config.modality(name)["input_size"]
            # ADR 0008 scalars are GMM-headed directly, over the same k as
            # every other modality's GMMHead.
            k = config.scalar_gmm_k if name in scalar_names else config.jetformer_gmm_k
            decoders[name] = GMMHead(config.hidden_size, input_size, k)
        self.decoders = nn.ModuleDict(decoders)

    def forward(
        self,
        hidden_states: torch.Tensor,  # [batch_size*seq_length, hidden]
        modality_masks: Dict[str, torch.Tensor],  # name -> bool [batch_size, seq_length]
    ):
        # hidden_states are flattened row-major, matching the collator's
        # concatenation order of modality_values — boolean indexing aligns
        # predictions with targets without explicit indices.
        out = {}
        for name, decoder in self.decoders.items():
            pred_positions = left_shift_mask(modality_masks[name]).view(-1)
            out[f"{name}_pred"] = decoder(hidden_states[pred_positions])
        return out


# --- ADR 0014 §2a: MFU accounting -------------------------------------------
# The backbone estimate alone flatters a 70M model with 47 modality heads, so
# the encoders, position embedders, decoders/GMM heads and jetformer flows are
# priced here. Everything below is FORWARD flops per token of that modality;
# callers multiply by 3 for fwd+bwd, matching qwen's get_flops convention.

# bf16 dense peak, no sparsity credit (ADR 0014 §2a). Override with
# $ASTROPT3_PEAK_TFLOPS on a device not listed here.
_PEAK_TFLOPS = {
    "A100": 312.0,
    "H100": 989.0,
    "H200": 989.0,
    "GH200": 989.0,
    "L40": 181.0,
}


def peak_tflops_per_gpu() -> Optional[float]:
    """Accelerator bf16 dense peak, for the MFU denominator."""
    override = os.environ.get("ASTROPT3_PEAK_TFLOPS")
    if override:
        return float(override)
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name()
    for key, value in _PEAK_TFLOPS.items():
        if key in name:
            return value
    return None


def modality_flops_per_token(config: AstroPT3Config) -> Dict[str, float]:
    """Forward FLOPs per loss-bearing token, per modality.

    Priced against the modules in this file: :class:`Encoder`,
    :class:`PositionEmbedder`, :class:`GMMHead`, and, for patch modalities,
    :class:`TinyFlow1D`. A linear layer costs ``2 * in * out``; an index
    position embedding is a lookup and costs nothing.

    ADR 0014 §3 requires this to be recomputed per model config — per-band
    changes both the token count and the head width, so arms must never
    share a pinned constant.
    """
    hidden = config.hidden_size
    per_token: Dict[str, float] = {}
    for name in config.modality_names():
        modality = config.modality(name)
        width = modality["input_size"]
        scalar = modality.get("scalar", False)

        flops = 2 * width * hidden  # Encoder: one Linear

        if modality.get("pos_type", "index") == "continuous":
            flops += 2 * modality.get("pos_input_size", 1) * hidden

        if scalar:
            k = config.scalar_gmm_k
            flops += 2 * hidden * k * (1 + 2 * width)
        else:
            k = config.jetformer_gmm_k
            flops += 2 * hidden * k * (1 + 2 * width)
            # TinyFlow1D: per coupling block, Linear(D/2 -> H) + Linear(H -> D)
            hidden_dim = config.jetformer_flow_hidden
            flops += config.jetformer_flow_steps * (
                2 * (width // 2) * hidden_dim + 2 * hidden_dim * width
            )

        per_token[name] = float(flops)
    return per_token


class AstroPT3Loss(nn.Module):
    """ADR 0013 family-balanced modality loss, in fp32.

    Present modalities are averaged within image/spectrum/scalar, then present
    family means are combined at 1:1:0.1. Absent modalities still contribute
    ``0 * pred.sum()`` to keep their decoders in the DDP graph.
    """

    def __init__(self, config: AstroPT3Config):
        super().__init__()
        self.loss_aggregation = config.loss_aggregation
        self.gmm_k = config.jetformer_gmm_k
        self.scalar_gmm_k = config.scalar_gmm_k
        self.scalar_names = {name for name in config.modality_names() if config.modality(name).get("scalar", False)}
        self.modality_dims = {name: config.modality(name)["input_size"] for name in config.modality_names()}
        self.modality_families = {name: config.modality(name)["family"] for name in config.modality_names()}
        self.loss_weights = {name: config.modality(name).get("loss_weight", 1.0) for name in config.modality_names()}

    def forward(
        self,
        modality_values: Dict[str, torch.Tensor],  # name -> [n_m, input_size] (scalar targets)
        **predictions: torch.Tensor,  # {name}_pred + {name}_z / {name}_logdet
    ) -> Dict[str, torch.Tensor]:
        graph_zero = None
        losses_by_family = {"image": [], "spectrum": [], "scalar": []}
        legacy_terms = []
        out = {}
        for name in self.modality_dims:
            pred = predictions[f"{name}_pred"]
            present = pred.shape[0] > 0
            if not present:
                mod_loss = pred.sum().float()  # 0.0, but keeps the decoder in the graph
            elif name in self.scalar_names:
                # ADR 0008: GMM NLL on the raw normalized scalar — no flow,
                # no logdet
                logits_pi, mu, log_sigma = unpack_gmm_params(pred.float(), self.scalar_gmm_k, self.modality_dims[name])
                mod_loss = gmm_nll(modality_values[name].float(), logits_pi, mu, log_sigma).mean()
            else:
                # exact patch-space likelihood: NLL_GMM(z) - logdet (can go
                # negative); z/logdet come clean from the embedding block
                logits_pi, mu, log_sigma = unpack_gmm_params(pred.float(), self.gmm_k, self.modality_dims[name])
                nll = gmm_nll(predictions[f"{name}_z"].float(), logits_pi, mu, log_sigma)
                mod_loss = (nll - predictions[f"{name}_logdet"].float()).mean()
            if present:
                losses_by_family[self.modality_families[name]].append(mod_loss)
                legacy_terms.append(self.loss_weights[name] * mod_loss)
            else:
                graph_zero = mod_loss if graph_zero is None else graph_zero + mod_loss
            out[f"{name}_loss"] = mod_loss
            reference_loss = mod_loss  # device/dtype for absent families

        total = graph_zero
        weight_sum = 0.0
        family_weights = {"image": 1.0, "spectrum": 1.0, "scalar": 0.1}
        for family, losses in losses_by_family.items():
            if not losses:
                # PipelineBlock asserts an exact output-key set, so a family
                # absent from THIS batch still has to report a key; it stays
                # out of the weighted total, and 0.0 reads as "no targets"
                # against the per-modality losses beside it
                out[f"{family}_family_loss"] = torch.zeros_like(reference_loss)
                continue
            family_loss = torch.stack(losses).mean()
            out[f"{family}_family_loss"] = family_loss
            weighted = family_weights[family] * family_loss
            total = weighted if total is None else total + weighted
            weight_sum += family_weights[family]
        if total is None:
            raise ValueError("AstroPT3 loss received no modality predictions")
        if self.loss_aggregation == "family":
            out["loss"] = total / weight_sum if weight_sum else total
        else:
            legacy_total = torch.stack(legacy_terms).sum() if legacy_terms else total
            if graph_zero is not None:
                legacy_total = legacy_total + graph_zero
            out["loss"] = legacy_total / max(len(legacy_terms), 1)
        return out


class AstroPT3Model(nn.Module):
    """Pipeline graph: embedding assembly -> Qwen2 decoder stack -> norm -> heads."""

    def __init__(
        self,
        config: AstroPT3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        self.p2p = P2P(parallel_context.pp_pg, device=torch.device("cuda"))
        self.config = config
        self.parallel_config = parallel_config
        self.parallel_context = parallel_context
        self.tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE

        # the embedding block additionally emits the clean latent {m}_z [n, D]
        # and {m}_logdet [n] for the loss (PP=1 is asserted, so these dict
        # outputs pass through the PipelineBlock locally). Scalar modalities
        # emit no z/logdet — they bypass the flow.
        patch_names = [name for name in config.modality_names() if not config.modality(name).get("scalar", False)]
        self.jet_keys = {f"{name}_z" for name in patch_names} | {f"{name}_logdet" for name in patch_names}

        self.token_position_embeddings = PipelineBlock(
            p2p=self.p2p,
            module_builder=AstroPT3Embedding,
            module_kwargs={
                "config": config,
                "parallel_config": parallel_config,
                "tp_pg": parallel_context.tp_pg,
                "random_states": random_states,
            },
            module_input_keys={"input_ids", "position_ids", "modality_values", "modality_positions", "modality_masks"},
            module_output_keys={"input_embeds", "position_ids"} | self.jet_keys,
        )

        self.decoder = nn.ModuleList(
            [
                PipelineBlock(
                    p2p=self.p2p,
                    module_builder=Qwen2DecoderLayer,
                    module_kwargs={
                        "config": config,
                        "parallel_config": parallel_config,
                        "tp_pg": parallel_context.tp_pg,
                        "cp_pg": parallel_context.cp_pg,
                        "layer_idx": layer_idx,
                    },
                    module_input_keys={"hidden_states", "position_ids", "cu_seqlens"},
                    module_output_keys={"hidden_states", "position_ids", "cu_seqlens"},
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.final_layer_norm = PipelineBlock(
            p2p=self.p2p,
            module_builder=TritonRMSNorm if config._fused_rms_norm else RMSNorm,
            module_kwargs={"hidden_size": config.hidden_size, "eps": config.rms_norm_eps},
            module_input_keys={"input"},
            module_output_keys={"hidden_states"},
        )

        self.modality_head = PipelineBlock(
            p2p=self.p2p,
            module_builder=AstroPT3ModalityHead,
            module_kwargs={"config": config},
            module_input_keys={"hidden_states", "modality_masks"},
            module_output_keys={f"{name}_pred" for name in config.modality_names()},
        )

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        modality_values: Dict[str, torch.Tensor],
        modality_positions: Dict[str, torch.Tensor],
        modality_masks: Dict[str, torch.Tensor],
    ):
        output = self.token_position_embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            modality_values=modality_values,
            modality_positions=modality_positions,
            modality_masks=modality_masks,
        )

        # Position restarts (including position-0 pads, each its own segment)
        # define the packed-document boundaries, exactly as upstream qwen.
        cu_seqlens = None
        if isinstance(position_ids, TensorPointer):
            raise ValueError("AstroPT3 requires PP=1 tensor inputs")
        if position_ids.numel() > 0:
            start_indices = torch.where(position_ids.view(-1) == 0)[0]
            cu_seqlens = torch.cat(
                [start_indices, torch.tensor([position_ids.numel()], dtype=torch.int32, device=start_indices.device)]
            ).to(torch.int32)

        decoder_states = {
            "hidden_states": output["input_embeds"],
            "position_ids": output["position_ids"],
            "cu_seqlens": cu_seqlens,
        }
        for decoder_layer in self.decoder:
            decoder_states = decoder_layer(**decoder_states)

        hidden_states = self.final_layer_norm(input=decoder_states["hidden_states"])["hidden_states"]

        predictions = self.modality_head(hidden_states=hidden_states, modality_masks=modality_masks)
        # jetformer: forward the embedding block's clean z/logdet to the loss
        return {**predictions, **{k: output[k] for k in self.jet_keys}}

    def get_block_compute_costs(self):
        """Compute costs per block for PP load balancing (PP=1: cosmetic)."""
        model_config = self.config
        d_ff = model_config.intermediate_size
        d_qkv = model_config.hidden_size // model_config.num_attention_heads
        head_cost = sum(2 * model_config.hidden_size * m["input_size"] for m in model_config.modalities or ())
        return {
            Qwen2DecoderLayer: 4 * model_config.num_attention_heads * d_qkv * model_config.hidden_size
            + 3 * d_ff * model_config.hidden_size,
            AstroPT3ModalityHead: head_cost,
        }

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        """Model/hardware FLOPs per second (vocab head term is the tiny 64-id one)."""
        world_size = self.parallel_context.world_pg.size()
        model_flops, hardware_flops = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )
        model_flops_per_s = model_flops / (iteration_time_in_sec * world_size * 1e12)
        hardware_flops_per_s = hardware_flops / (iteration_time_in_sec * world_size * 1e12)
        return model_flops_per_s, hardware_flops_per_s

    def get_mfu_report(self, iteration_time_in_sec, sequence_length, global_batch_size, telemetry):
        """ADR 0014 §2a: MFU, decomposed into the three factors that move it.

            MFU = MFU_busy x (1 - stall_share) x utilisation_packing

        The decomposition is the point. An undecomposed comparison between a
        fused and a per-band arm confounds a data-pipeline effect
        (``stall_share``) with a model-shape effect (``MFU_busy``) and
        supports the wrong conclusion, which is why §11 refuses headline MFU
        as an acceptance criterion.

        ``telemetry`` is a drained ``astropt3.data.telemetry`` step record:
        non-padding tokens, per-modality loss-bearing tokens, and the
        main-process loader wait. Returns {} when it is absent (telemetry off)
        or the accelerator peak is unknown.
        """
        world_size = self.parallel_context.world_pg.size()
        if not telemetry or not telemetry.get("tokens_total"):
            return {}
        peak = peak_tflops_per_gpu()
        if peak is None:
            return {}

        utilisation = telemetry["utilisation_packing"]
        backbone, _ = get_flops(
            num_layers=self.config.num_hidden_layers,
            hidden_size=self.config.hidden_size,
            num_heads=self.config.num_attention_heads,
            num_key_value_heads=self.config.num_key_value_heads,
            vocab_size=self.config.vocab_size,
            ffn_hidden_size=self.config.intermediate_size,
            seq_len=sequence_length,
            batch_size=global_batch_size,
        )
        # padding earns no MFU credit (§2a): a padded-out packed row is wasted
        # compute exactly as a stall is wasted time.
        # ponytail: linear scaling, though the attention term is quadratic in
        # seq_len. It errs toward OVER-counting (document masking already
        # makes real attention block-diagonal and cheaper than full seq^2), so
        # reported MFU is an upper bound. Model the block structure only if an
        # arm is ever decided on the attention term alone.
        backbone *= utilisation

        per_token = modality_flops_per_token(self.config)
        modality = 3 * sum(  # 1 fwd + 2 bwd, matching get_flops
            count * per_token.get(name, 0.0)
            for name, count in telemetry.get("loss_tokens", {}).items()
        )
        # telemetry counts this DP rank's micro-batches; the backbone estimate
        # is already global, so scale the modality term the same way
        modality *= world_size

        total_flops = backbone + modality
        elapsed = max(iteration_time_in_sec, 1e-9)
        busy = max(elapsed - telemetry.get("loader_wait_s", 0.0), 1e-9)
        stall_share = 1.0 - busy / elapsed
        denominator = peak * 1e12 * world_size

        loss_tokens = sum(telemetry.get("loss_tokens", {}).values()) or 1
        return {
            # step_seconds and model_flops are absolute so the offline report
            # can time-weight them. A mean of per-step MFU (or of per-step
            # stall_share) is NOT the run's MFU: this corpus is bimodal —
            # most steps stall for nothing and a few stall for a minute — so
            # averaging ratios buries exactly the behaviour being measured.
            "step_seconds": elapsed,
            "model_flops": total_flops,
            "mfu": total_flops / (elapsed * denominator),
            "mfu_busy": total_flops / (busy * denominator),
            "stall_share": stall_share,
            "utilisation_packing": utilisation,
            "flops_per_token": total_flops / (3 * loss_tokens * world_size),
            # deliberately NOT "model_tflops_per_gpu": nanotron logs its own
            # backbone-only, padding-credited number under that name, and two
            # LogItems with one key silently collide in wandb. This one counts
            # the modality heads and refuses padding credit, so it reads lower.
            "astropt3_tflops_per_gpu": total_flops / (elapsed * world_size * 1e12),
            "peak_tflops_per_gpu": peak,
            "loader_wait_s": telemetry.get("loader_wait_s", 0.0),
        }


class AstroPT3ForTraining(NanotronModel, LoggingCollectorMixin):
    def __init__(
        self,
        config: AstroPT3Config,
        parallel_context: ParallelContext,
        parallel_config: Optional[ParallelismArgs],
        random_states: Optional[RandomStates] = None,
    ):
        super().__init__()
        assert parallel_context.pp_pg.size() == 1, "astropt3 is PP=1 by design (see astro/PLAN.md)"
        tp_mode = parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE
        assert tp_mode is TensorParallelLinearMode.ALL_REDUCE, (
            "astropt3 keeps modality encoders/decoders replicated across TP, which requires the "
            "hidden stream to be replicated at the embedding and head blocks (tp_mode: ALL_REDUCE). "
            "REDUCE_SCATTER shards the sequence across TP ranks and is not supported."
        )
        self.model = AstroPT3Model(
            config=config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=random_states,
        )
        self.loss = PipelineBlock(
            p2p=self.model.p2p,
            module_builder=AstroPT3Loss,
            module_kwargs={"config": config},
            module_input_keys={"modality_values"}
            | {f"{name}_pred" for name in config.modality_names()}
            | self.model.jet_keys,
            module_output_keys={"loss"}
            | {f"{name}_loss" for name in config.modality_names()}
            # families are the fixed image/spectrum/scalar set (config validates
            # it), and the loss emits all three every batch to keep this static
            | {f"{family}_family_loss" for family in ("image", "spectrum", "scalar")},
        )
        self.parallel_context = parallel_context
        self.config = config
        self.parallel_config = parallel_config

    def forward(
        self,
        input_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        position_ids: Union[torch.Tensor, TensorPointer],  # [batch_size, seq_length]
        **modality_tensors: Union[torch.Tensor, TensorPointer],  # {m}_values / {m}_positions / {m}_mask
    ) -> Dict[str, Union[torch.Tensor, TensorPointer]]:
        names = cast(AstroPT3Config, self.config).modality_names()
        modality_values = {name: modality_tensors[f"{name}_values"] for name in names}
        modality_positions = {name: modality_tensors[f"{name}_positions"] for name in names}
        modality_masks = {name: modality_tensors[f"{name}_mask"] for name in names}

        predictions = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            modality_values=modality_values,
            modality_positions=modality_positions,
            modality_masks=modality_masks,
        )
        return self.loss(modality_values=modality_values, **predictions)

    def set_jet_noise_frac(self, frac: float):
        """Drive the jetformer noise curriculum."""
        module = getattr(self.model.token_position_embeddings, "pp_block", None)
        if module is not None:
            module.set_jet_noise_frac(frac)

    @torch.no_grad()
    def init_model_randomly(self, config: Config):
        """Stock nanotron init, extended to the replicated modality modules.

        Plain ``nn.Linear``/``nn.Embedding`` (modality encoders, decoders,
        position embedders) reuse the parametrizator's column-linear and
        embedding rules — the same normal(0, std) the HF side gets from
        ``_init_weights``. Cross-TP consistency comes from the tied-parameter
        sync that runs right after init.
        """
        init_method = config.model.init_method
        if isinstance(init_method, RandomInit):
            parametrizator_cls = StandardParametrizator
        elif isinstance(init_method, SpectralMupInit):
            parametrizator_cls = SpectralMupParametrizator
        else:
            raise ValueError(f"Unknown init method {init_method}")

        parametrizator = parametrizator_cls(config=config)  # type: ignore[arg-type]
        parametrizator.MODULE_TO_PARAMETRIZE[nn.Linear] = parametrizator.MODULE_TO_PARAMETRIZE[
            TensorParallelColumnLinear
        ]
        parametrizator.MODULE_TO_PARAMETRIZE[nn.Embedding] = parametrizator.MODULE_TO_PARAMETRIZE[
            TensorParallelEmbedding
        ]

        log_rank(
            f"Parametrizing model parameters using {parametrizator.__class__.__name__}",
            logger=logger,
            level=logging.INFO,
            rank=0,
        )

        model = self
        initialized_parameters = set()
        module_id_to_prefix = {id(module): f"{module_name}." for module_name, module in model.named_modules()}
        module_id_to_prefix[id(model)] = ""

        for param_name, param in model.named_parameters():
            assert isinstance(param, NanotronParameter)

            module_name, param_name = param_name.rsplit(".", 1)

            if param.is_tied:
                tied_info = param.get_tied_info()
                full_param_name = tied_info.get_full_name_from_module_id_to_prefix(
                    module_id_to_prefix=module_id_to_prefix
                )
            else:
                full_param_name = f"{module_name}.{param_name}"

            if full_param_name in initialized_parameters:
                continue

            module = model.get_submodule(module_name)
            parametrizator.parametrize(param_name, module)

            assert full_param_name not in initialized_parameters
            initialized_parameters.add(full_param_name)

        expected_parameters = set()
        for name, parameter in model.named_parameters():
            param = cast(NanotronParameter, parameter)
            expected_parameters.add(
                param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
                if param.is_tied
                else name
            )
        assert initialized_parameters == expected_parameters, (
            "Somehow the initialized set of parameters don't match:\n"
            f" - Expected: {expected_parameters}\n - Got: {initialized_parameters}"
        )

    def get_embeddings_lm_head_tied_names(self):
        return []  # no lm_head to tie

    def get_block_compute_costs(self):
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)

    def get_mfu_report(self, iteration_time_in_sec, sequence_length, global_batch_size, telemetry):
        return self.model.get_mfu_report(
            iteration_time_in_sec, sequence_length, global_batch_size, telemetry
        )
