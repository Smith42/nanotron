"""AstroPT3 training model: SmolLM3 (qwen2-style) decoder body with
continuous-modality inputs and per-modality regression heads.

This mirrors the HF release implementation in the ``astropt3`` package
(``astro/src/astropt3/modeling_astropt3.py``), replacing exactly two blocks of
the upstream Qwen2 pipeline graph:

- the vocab ``TensorParallelEmbedding`` block becomes :class:`AstroPT3Embedding`:
  a 64-id special-token embedding plus additive per-modality deltas
  ``encoder_m(value) + pos_embed_m(position)`` at placeholder slots;
- the ``lm_head`` + sharded-CE ``Loss`` blocks become :class:`AstroPT3ModalityHead`
  (per-modality affine decoders applied one position LEFT of each modality
  token — astroPT's ``starts-1`` alignment) plus :class:`AstroPT3Loss`
  (``loss_weight``-weighted mean of per-modality Huber losses).

Parallelism contract (see astro/PLAN.md):

- **PP=1 always** (asserted): the whole micro-batch dict reaches every rank,
  so dict-valued inputs pass through PipelineBlocks locally and modality
  tensors never cross pipeline stages.
- **TP**: the transformer body is sharded as upstream. Modality
  encoders/decoders/pos-embedders are tiny affine layers kept **replicated**
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
from typing import Dict, Optional, Union

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
# implementations, one weight source of truth). Attribute names (c_fc,
# c_proj, embed) are part of the conversion contract in
# tools/astropt3/convert_weights.py — keep them in sync.


class Encoder(nn.Module):
    """Data space -> embedding space. Replicated across TP."""

    def __init__(self, hidden_size: int, in_size: int, tokeniser: str = "affine", bias: bool = False):
        super().__init__()
        self.tokeniser = tokeniser
        if tokeniser in ("affine", "jetformer"):
            self.c_fc = nn.Linear(in_size, hidden_size, bias=bias)
        elif tokeniser == "aim":
            self.c_fc = nn.Linear(in_size, 4 * hidden_size, bias=bias)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * hidden_size, hidden_size, bias=bias)
        else:
            raise ValueError(f"unknown tokeniser {tokeniser!r}")

    def forward(self, x):
        if self.tokeniser in ("affine", "jetformer"):
            return self.c_fc(x)
        return self.c_proj(self.gelu(self.c_fc(x)))


class Decoder(nn.Module):
    """Embedding space -> data space (regression head). Replicated across TP."""

    def __init__(self, hidden_size: int, out_size: int, tokeniser: str = "affine", bias: bool = False):
        super().__init__()
        self.tokeniser = tokeniser
        if tokeniser == "affine":
            self.c_fc = nn.Linear(hidden_size, out_size, bias=bias)
        elif tokeniser == "aim":
            self.c_fc = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * hidden_size, out_size, bias=bias)
        else:
            raise ValueError(f"unknown tokeniser {tokeniser!r}")

    def forward(self, x):
        if self.tokeniser == "affine":
            return self.c_fc(x)
        return self.c_proj(self.gelu(self.c_fc(x)))


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


# --- "jetformer" tokeniser modules ------------------------------------------
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
        self.token_embedding = TensorParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            padding_idx=config.pad_token_id,
            pg=tp_pg,
            mode=parallel_config.tp_mode if parallel_config is not None else TensorParallelLinearMode.ALL_REDUCE,
        )
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(config.hidden_size, config.modality(name)["input_size"], config.tokeniser)
                for name in config.modality_names()
            }
        )
        self.pos_embeds = nn.ModuleDict(
            {name: PositionEmbedder(config.hidden_size, config.modality(name)) for name in config.modality_names()}
        )
        self.tokeniser = config.tokeniser
        if config.tokeniser == "jetformer":
            self.flows = nn.ModuleDict(
                {
                    name: TinyFlow1D(
                        config.modality(name)["input_size"],
                        steps=config.jetformer_flow_steps,
                        hidden_dim=config.jetformer_flow_hidden,
                    )
                    for name in config.modality_names()
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
        self.jet_noise_frac = float(min(max(frac, 0.0), 1.0))

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
            if self.tokeniser == "jetformer":
                z, logdet = self.flows[name](values)
                extras[f"{name}_z"] = z
                extras[f"{name}_logdet"] = logdet
                sigma = self.jetformer_noise_max + (
                    self.jetformer_noise_min - self.jetformer_noise_max
                ) * self.jet_noise_frac
                values = z
                if self.training and sigma > 0:
                    # the noise must be IDENTICAL across TP ranks or the
                    # replicated-hidden-stream contract breaks — draw it under
                    # the trainer's tp_synced random state
                    synced = self.random_states is not None and "tp_synced" in self.random_states
                    with branch_random_state(self.random_states, "tp_synced", enabled=synced):
                        values = z + sigma * torch.randn_like(z)
            content = encoder(values) + self.pos_embeds[name](modality_positions[name]).to(input_embeds.dtype)
            delta = delta.index_put((modality_masks[name].view(-1),), content.to(input_embeds.dtype))
        return {"input_embeds": input_embeds + delta, "position_ids": position_ids, **extras}


class AstroPT3ModalityHead(nn.Module):
    """Per-modality regression decoders at ``starts-1``-aligned positions."""

    def __init__(self, config: AstroPT3Config):
        super().__init__()
        if config.tokeniser == "jetformer":
            # GMM heads under the same name/key ({m}_pred stays one tensor,
            # the raw K*(1+2D) projection unpacked inside the loss)
            self.decoders = nn.ModuleDict(
                {
                    name: GMMHead(
                        config.hidden_size,
                        config.modality(name)["input_size"],
                        config.jetformer_gmm_k,
                    )
                    for name in config.modality_names()
                }
            )
        else:
            self.decoders = nn.ModuleDict(
                {
                    name: Decoder(config.hidden_size, config.modality(name)["input_size"], config.tokeniser)
                    for name in config.modality_names()
                }
            )

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


class AstroPT3Loss(nn.Module):
    """Weighted mean of per-modality Huber losses (astroPT semantics).

    Mirrors the HF side: each present modality contributes
    ``loss_weight * huber(pred, target)``; the total is divided by the number
    of modalities present in the micro-batch. Absent modalities contribute
    ``0 * pred.sum()`` to keep their decoder in the graph for DDP. Loss math
    runs in fp32.
    """

    def __init__(self, config: AstroPT3Config):
        super().__init__()
        self.tokeniser = config.tokeniser
        self.huber_delta = config.huber_delta
        self.gmm_k = config.jetformer_gmm_k
        self.modality_dims = {name: config.modality(name)["input_size"] for name in config.modality_names()}
        self.loss_weights = {name: config.modality(name).get("loss_weight", 1.0) for name in config.modality_names()}

    def forward(
        self,
        modality_values: Dict[str, torch.Tensor],  # name -> [n_m, input_size] (affine targets)
        **predictions: torch.Tensor,  # {name}_pred [+ jetformer {name}_z / {name}_logdet]
    ) -> Dict[str, torch.Tensor]:
        total = None
        n_present = 0
        out = {}
        for name, weight in self.loss_weights.items():
            pred = predictions[f"{name}_pred"]
            if pred.shape[0] == 0:
                mod_loss = pred.sum().float()  # 0.0, but keeps the decoder in the graph
            elif self.tokeniser == "jetformer":
                # exact patch-space likelihood: NLL_GMM(z) - logdet (can go
                # negative); z/logdet come clean from the embedding block
                logits_pi, mu, log_sigma = unpack_gmm_params(
                    pred.float(), self.gmm_k, self.modality_dims[name]
                )
                nll = gmm_nll(predictions[f"{name}_z"].float(), logits_pi, mu, log_sigma)
                mod_loss = (nll - predictions[f"{name}_logdet"].float()).mean()
                n_present += 1
            else:
                target = modality_values[name]
                mod_loss = F.huber_loss(pred.float(), target.float(), delta=self.huber_delta)
                n_present += 1
            total = mod_loss * weight if total is None else total + mod_loss * weight
            out[f"{name}_loss"] = mod_loss
        out["loss"] = total / max(n_present, 1)
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

        # jetformer: the embedding block additionally emits the clean latent
        # {m}_z [n, D] and {m}_logdet [n] for the loss (PP=1 is asserted, so
        # these dict outputs pass through the PipelineBlock locally)
        jet_keys = set()
        if config.tokeniser == "jetformer":
            jet_keys = {f"{name}_z" for name in config.modality_names()} | {
                f"{name}_logdet" for name in config.modality_names()
            }
        self.jet_keys = jet_keys

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
            module_output_keys={"input_embeds", "position_ids"} | jet_keys,
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
        head_cost = sum(
            2 * model_config.hidden_size * m["input_size"] for m in model_config.modalities
        )
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
            module_output_keys={"loss"} | {f"{name}_loss" for name in config.modality_names()},
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
        names = self.config.modality_names()
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
        """Drive the jetformer noise curriculum (no-op for other tokenisers)."""
        module = getattr(self.model.token_position_embeddings, "pp_block", None)
        if module is not None and hasattr(module, "set_jet_noise_frac"):
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

        parametrizator = parametrizator_cls(config=config)
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

        assert initialized_parameters == {
            param.get_tied_info().get_full_name_from_module_id_to_prefix(module_id_to_prefix=module_id_to_prefix)
            if param.is_tied
            else name
            for name, param in model.named_parameters()
        }, f"Somehow the initialized set of parameters don't match:\n - Expected: { {name for name, _ in model.named_parameters()} }\n - Got: {initialized_parameters}"

    def get_embeddings_lm_head_tied_names(self):
        return []  # no lm_head to tie

    def get_block_compute_costs(self):
        return self.model.get_block_compute_costs()

    def get_flops_per_sec(self, iteration_time_in_sec, sequence_length, global_batch_size):
        return self.model.get_flops_per_sec(iteration_time_in_sec, sequence_length, global_batch_size)
