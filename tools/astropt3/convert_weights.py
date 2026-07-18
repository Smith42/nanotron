"""Shared helpers for AstroPT3 nanotron <-> HF conversion.

Weight-name contract (modeled on examples/smollm3):

- transformer body: identical to the SmolLM3 mapping, minus the lm_head;
- modality modules (attribute names are shared between the two
  implementations by design):

  =====================  =====================================================
  HF (astropt3 package)  nanotron (models/astropt3.py)
  =====================  =====================================================
  model.embed_tokens.*   model.token_position_embeddings.pp_block.token_embedding.*
  encoders.{m}.*         model.token_position_embeddings.pp_block.encoders.{m}.*
  pos_embeds.{m}.*       model.token_position_embeddings.pp_block.pos_embeds.{m}.*
  decoders.{m}.*         model.modality_head.pp_block.decoders.{m}.*
  flows.{m}.*            model.token_position_embeddings.pp_block.flows.{m}.*  (jetformer)
  model.norm.weight      model.final_layer_norm.pp_block.weight
  =====================  =====================================================

Run these scripts in the ``[train]`` environment (nanotron + astropt3 + a
GPU): the nanotron model imports flash-attn.
"""

import json
from pathlib import Path
from typing import Optional

import nanotron
import torch
from nanotron.config import AstroPT3Config as NanotronAstroPT3Config
from nanotron.config import (
    OneForwardOneBackwardPipelineEngine,
    ParallelismArgs,
    PipelineEngine,
    TensorParallelLinearMode,
)
from nanotron.models.astropt3 import AstroPT3ForTraining
from nanotron.trainer import mark_tied_parameters

# HF-side parameter prefix -> nanotron parameter prefix, applied to the
# modality modules whatever their inner structure (affine or aim tokeniser).
_MODALITY_PREFIX_MAP = {
    "encoders.": "model.token_position_embeddings.pp_block.encoders.",
    "pos_embeds.": "model.token_position_embeddings.pp_block.pos_embeds.",
    "decoders.": "model.modality_head.pp_block.decoders.",
}


def get_weight_mapping(config: NanotronAstroPT3Config, nt_to_hf: bool = True) -> dict:
    """nanotron<->HF parameter name mapping (see examples/smollm3)."""
    hf_to_nt_map = {}
    hf_to_nt_map["model.embed_tokens.weight"] = "model.token_position_embeddings.pp_block.token_embedding.weight"
    hf_to_nt_map["model.norm.weight"] = "model.final_layer_norm.pp_block.weight"

    for i in range(config.num_hidden_layers):
        hf_prefix = f"model.layers.{i}"
        nt_prefix = f"model.decoder.{i}.pp_block"
        hf_to_nt_map[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{nt_prefix}.attn.qkv_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{nt_prefix}.attn.qkv_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{nt_prefix}.attn.qkv_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.self_attn.o_proj.weight"] = f"{nt_prefix}.attn.o_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.mlp.gate_proj.weight"] = f"{nt_prefix}.mlp.gate_up_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.mlp.up_proj.weight"] = f"{nt_prefix}.mlp.gate_up_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.mlp.down_proj.weight"] = f"{nt_prefix}.mlp.down_proj.weight"
        hf_to_nt_map[f"{hf_prefix}.input_layernorm.weight"] = f"{nt_prefix}.input_layernorm.weight"
        hf_to_nt_map[f"{hf_prefix}.post_attention_layernorm.weight"] = f"{nt_prefix}.post_attention_layernorm.weight"

    # modality modules: generate from the tokeniser structure so affine, aim
    # and jetformer all map (weights only; all implementations use bias=False)
    if config.tokeniser == "jetformer":
        encoder_weights, decoder_weights = ["c_fc.weight"], ["proj.weight"]
    elif config.tokeniser == "affine":
        encoder_weights = decoder_weights = ["c_fc.weight"]
    else:  # aim
        encoder_weights = decoder_weights = ["c_fc.weight", "c_proj.weight"]
    for mod in config.modalities:
        name = mod["name"]
        for w in encoder_weights:
            hf_to_nt_map[f"encoders.{name}.{w}"] = f"{_MODALITY_PREFIX_MAP['encoders.']}{name}.{w}"
        for w in decoder_weights:
            hf_to_nt_map[f"decoders.{name}.{w}"] = f"{_MODALITY_PREFIX_MAP['decoders.']}{name}.{w}"
        hf_to_nt_map[f"pos_embeds.{name}.embed.weight"] = f"{_MODALITY_PREFIX_MAP['pos_embeds.']}{name}.embed.weight"
        if config.tokeniser == "jetformer":
            # HF flows.{m}.blocks.{i}.net.{0,2}.{weight,bias} <-> fork
            # ...token_position_embeddings.pp_block.flows.{m}... (CouplingMLP
            # nets carry biases, unlike the affine modality layers)
            for i in range(config.jetformer_flow_steps):
                for layer in (0, 2):
                    for p in ("weight", "bias"):
                        hf_key = f"flows.{name}.blocks.{i}.net.{layer}.{p}"
                        hf_to_nt_map[hf_key] = (
                            f"model.token_position_embeddings.pp_block.flows.{name}.blocks.{i}.net.{layer}.{p}"
                        )

    if nt_to_hf:
        nt_to_hf_map = {}
        for hf, nt in hf_to_nt_map.items():
            # qkv (and gate_up) fold several HF params into one nanotron param
            if nt in nt_to_hf_map and isinstance(nt_to_hf_map[nt], list):
                nt_to_hf_map[nt].append(hf)
            elif nt in nt_to_hf_map:
                nt_to_hf_map[nt] = [nt_to_hf_map[nt], hf]
            else:
                nt_to_hf_map[nt] = hf
        return nt_to_hf_map
    return hf_to_nt_map


def get_config_mapping(nt_to_hf: bool = True) -> dict:
    """Field mapping between the two AstroPT3Config implementations.

    ``no_rope_layer`` (nanotron) <-> ``no_rope_layer_interval`` (HF) is the
    one renamed field.
    """
    hf_to_nt_map = {
        "attention_bias": "attention_bias",
        "bos_token_id": "bos_token_id",
        "eos_token_id": "eos_token_id",
        "hidden_act": "hidden_act",
        "hidden_size": "hidden_size",
        "huber_delta": "huber_delta",
        "image_norm_divisor": "image_norm_divisor",
        "initializer_range": "initializer_range",
        "intermediate_size": "intermediate_size",
        "jetformer_flow_steps": "jetformer_flow_steps",
        "jetformer_flow_hidden": "jetformer_flow_hidden",
        "jetformer_gmm_k": "jetformer_gmm_k",
        "jetformer_noise_max": "jetformer_noise_max",
        "jetformer_noise_min": "jetformer_noise_min",
        "max_position_embeddings": "max_position_embeddings",
        "modalities": "modalities",
        "no_rope_layer_interval": "no_rope_layer",
        "num_attention_heads": "num_attention_heads",
        "num_hidden_layers": "num_hidden_layers",
        "num_key_value_heads": "num_key_value_heads",
        "pad_token_id": "pad_token_id",
        "rms_norm_eps": "rms_norm_eps",
        "rope_theta": "rope_theta",
        "shuffle_modality_order": "shuffle_modality_order",
        "spectra_norm_divisor": "spectra_norm_divisor",
        "spiral": "spiral",
        "tie_word_embeddings": "tie_word_embeddings",
        "tokeniser": "tokeniser",
        "vocab_size": "vocab_size",
    }
    if nt_to_hf:
        return {nt: hf for hf, nt in hf_to_nt_map.items()}
    return hf_to_nt_map


def make_parallel_config(
    dp: int = 1,
    pp: int = 1,
    tp: int = 1,
    pp_engine: PipelineEngine = OneForwardOneBackwardPipelineEngine(),
):
    # astropt3 requires ALL_REDUCE (replicated modality modules); async
    # column-linear communication is a REDUCE_SCATTER-only optimization.
    return ParallelismArgs(
        dp=dp,
        pp=pp,
        tp=tp,
        pp_engine=pp_engine,
        tp_mode=TensorParallelLinearMode.ALL_REDUCE,
        tp_linear_async_communication=False,
    )


def load_nanotron_model(
    model_config: Optional[NanotronAstroPT3Config] = None,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: Optional[Path] = None,
) -> AstroPT3ForTraining:
    """Build an AstroPT3ForTraining (TP=PP=DP=1) and optionally load weights."""
    if model_config is None:
        assert checkpoint_path is not None
        with open(checkpoint_path / "model_config.json") as f:
            model_config = NanotronAstroPT3Config(**json.load(f))
    parallel_config = make_parallel_config()
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=parallel_config.dp,
        pipeline_parallel_size=parallel_config.pp,
        tensor_parallel_size=parallel_config.tp,
    )
    nanotron_model = nanotron.models.build_model(
        model_builder=lambda: AstroPT3ForTraining(
            config=model_config,
            parallel_context=parallel_context,
            parallel_config=parallel_config,
            random_states=None,
        ),
        parallel_context=parallel_context,
        dtype=dtype,
        device=device,
    )
    mark_tied_parameters(model=nanotron_model, parallel_context=parallel_context)
    if checkpoint_path is not None:
        nanotron.serialize.load_weights(
            model=nanotron_model, parallel_context=parallel_context, root_folder=checkpoint_path
        )
    return nanotron_model
