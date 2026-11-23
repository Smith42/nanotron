"""AstroPT3 model and streaming dataset configuration.

The shared modality schema lives in ``astropt3.configuration_astropt3``.
Named modalities own fixed three-ID blocks. Configs must size ``vocab_size``
to cover their active IDs; there is no text vocabulary and no LM head.
"""

from dataclasses import dataclass
from typing import List, Optional

from nanotron.config.models_config import Qwen2Config


@dataclass
class AstroPT3Config(Qwen2Config):
    """Qwen2/SmolLM3 body + per-modality regression heads (no lm_head).

    ``is_astropt3_config`` is the yaml/python dispatch marker (see
    ``ModelArgs.__post_init__``), like ``is_qwen2_config`` upstream.
    """

    is_astropt3_config: bool = True
    modalities: Optional[List[dict]] = None
    eos_token_id: int = 1
    vocab_size: int = 169
    tie_word_embeddings: bool = False
    # jetformer regression head (mirrors the HF-side AstroPT3Config defaults):
    # per-modality JetFormerFlow + GMMHead, loss = mean(NLL_GMM(z) - logdet).
    # noise_max -> noise_min is the flow-stability curriculum, annealed by the
    # trainer via set_jet_noise_frac(iteration / train_steps).
    jetformer_flow_steps: int = 8
    jetformer_flow_hidden: int = 128
    jetformer_flow_depth: int = 4
    jetformer_flow_heads: int = 8
    jetformer_gmm_k: int = 4
    jetformer_noise_max: float = 0.1
    jetformer_noise_min: float = 0.0
    # ADR 0008 scalar modalities: mixture count of the scalar GMM heads
    # (carried into converted HF checkpoints)
    scalar_gmm_k: int = 5
    # arcsinh divisor (nMgy) of the physical image normalization; threaded
    # into the sequencer by astro's build_astropt3_dataloader and carried
    # into converted HF checkpoints (mirrors the HF-side default)
    image_norm_divisor: float = 0.01
    # arcsinh knee (nMgy) of the physical spectra normalization, threaded
    # and converted like image_norm_divisor.
    spectra_norm_divisor: float = 10.0
    # Center-outward image patch order (ADR 0004), carried into checkpoints.
    spiral: bool = True

    def __post_init__(self):
        # Standalone Nanotron configs for other models don't need astropt3.
        from astropt3.configuration_astropt3 import complete_modalities
        from astropt3.tokenization import required_vocab_size

        # Qwen2Config asserts num_hidden_layers % no_rope_layer == 0, but the
        # runtime rule is per-layer ((layer_idx+1) % no_rope_layer != 0 -> RoPE),
        # identical to HF SmolLM3's no_rope_layer_interval which allows any
        # layer count (e.g. the 23-layer 70M size). Bypass the assert only.
        no_rope_layer = self.no_rope_layer
        self.no_rope_layer = None
        super().__post_init__()
        self.no_rope_layer = no_rope_layer
        for field in ("jetformer_flow_steps", "jetformer_flow_hidden", "jetformer_flow_depth", "jetformer_flow_heads"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.jetformer_flow_hidden % self.jetformer_flow_heads:
            raise ValueError("jetformer_flow_hidden must be divisible by jetformer_flow_heads")
        self.modalities = complete_modalities(self.modalities)
        required_vocab = required_vocab_size(self.modalities)
        if self.vocab_size < required_vocab:
            raise ValueError(f"vocab_size={self.vocab_size} cannot hold modality token id {required_vocab - 1}")
        if self.tie_word_embeddings:
            raise ValueError("astropt3 has no lm_head to tie")

    def modality_names(self) -> List[str]:
        """Alphabetical registry order — fixes sequence order everywhere."""
        return sorted(modality["name"] for modality in self.modalities or ())

    def modality(self, name: str) -> dict:
        return next(modality for modality in self.modalities or () if modality["name"] == name)


@dataclass
class AstroPT3StreamingDatasetsArgs:
    """Explicit YAML-configured LSDB catalog streams.

    ``catalog_stream`` is required. Validation/decoding lives in ``astropt3``.
    DataLoader workers provide concurrency, without a resumable stream cursor.
    """

    is_astropt3_streaming: bool = True
    # Total optimizer-step exposures per batch from each loaded draw.
    echo_factor: int = 1
    catalog_stream: Optional[dict] = None
    paired_echo_factor: Optional[int] = None

    def __post_init__(self):
        if type(self.echo_factor) is not int or self.echo_factor < 1:
            raise ValueError("echo_factor must be a positive integer")
        if self.paired_echo_factor is not None and (
            type(self.paired_echo_factor) is not int or self.paired_echo_factor < 1
        ):
            raise ValueError("paired_echo_factor must be a positive integer")
        if not isinstance(self.catalog_stream, dict) or "catalogs" not in self.catalog_stream:
            raise ValueError("catalog_stream.catalogs is required; migrate legacy dataset YAML")
