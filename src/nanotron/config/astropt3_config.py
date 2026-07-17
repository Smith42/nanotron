"""AstroPT3 config: a Qwen2/SmolLM3 body driven by continuous-modality inputs.

Two additions over Qwen2Config:

- ``modalities``: list of per-modality dicts (name/input_size/patch_size/
  pos_type/pos_input_size/max_positions/loss_weight) mirroring the HF-side
  ``astropt3.configuration_astropt3.DEFAULT_MODALITIES``. Registry order is
  alphabetical by name everywhere.
- ``AstroPT3StreamingDatasetsArgs``: the ``astropt3_streaming`` dataset type
  consumed by ``run_train.py`` (packed multimodal micro-batches built by the
  ``astropt3`` package's ``data/nanotron_loader.py``).

The special-token vocabulary is frozen at 64 ids (see the astropt3 package's
``tokenization.py``); there is no text vocab and no lm_head.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from nanotron.config.models_config import Qwen2Config

# Pinned to the verified MMU pilot schemas (images (3,152,152) patch 8;
# DESI spectra 7781 bins patch 256). Must stay in sync with the HF-side
# DEFAULT_MODALITIES in astropt3/configuration_astropt3.py.
DEFAULT_MODALITIES = [
    {
        "name": "images",
        "input_size": 192,
        "patch_size": 8,
        "pos_type": "index",
        "pos_input_size": 1,
        "max_positions": 361,
        "loss_weight": 1.0,
    },
    {
        "name": "spectra",
        "input_size": 256,
        "patch_size": 256,
        "pos_type": "continuous",
        "pos_input_size": 1,
        "max_positions": 1024,
        "loss_weight": 1.0,
    },
]


@dataclass
class AstroPT3Config(Qwen2Config):
    """Qwen2/SmolLM3 body + per-modality regression heads (no lm_head).

    ``is_astropt3_config`` is the yaml/python dispatch marker (see
    ``ModelArgs.__post_init__``), like ``is_qwen2_config`` upstream.
    """

    is_astropt3_config: bool = True
    modalities: Optional[List[dict]] = None
    tokeniser: str = "affine"  # "affine" (default), "aim" (astroPT MLP) or "jetformer" (flow + GMM)
    huber_delta: float = 1.0
    vocab_size: int = 64
    tie_word_embeddings: bool = False
    # jetformer tokeniser (mirrors the HF-side AstroPT3Config defaults):
    # per-modality TinyFlow1D + GMMHead, loss = mean(NLL_GMM(z) - logdet).
    # noise_max -> noise_min is the flow-stability curriculum, annealed by the
    # trainer via set_jet_noise_frac(iteration / train_steps).
    jetformer_flow_steps: int = 4
    jetformer_flow_hidden: int = 128
    jetformer_gmm_k: int = 4
    jetformer_noise_max: float = 0.1
    jetformer_noise_min: float = 0.0
    # arcsinh divisor (nMgy) of the physical image normalization; threaded
    # into the sequencer by astro's build_astropt3_dataloader and carried
    # into converted HF checkpoints (mirrors the HF-side default)
    image_norm_divisor: float = 0.01
    # center-outward spiral image patch order (ADR 0004); threaded into the
    # sequencer like image_norm_divisor and carried into converted HF
    # checkpoints. Default True matching the HF-side AstroPT3Config (the
    # agreed going-forward default); raster checkpoints must set
    # spiral: false explicitly.
    spiral: bool = True

    def __post_init__(self):
        # Qwen2Config asserts num_hidden_layers % no_rope_layer == 0, but the
        # runtime rule is per-layer ((layer_idx+1) % no_rope_layer != 0 -> RoPE),
        # identical to HF SmolLM3's no_rope_layer_interval which allows any
        # layer count (e.g. the 23-layer 70M size). Bypass the assert only.
        no_rope_layer = self.no_rope_layer
        self.no_rope_layer = None
        super().__post_init__()
        self.no_rope_layer = no_rope_layer
        if self.modalities is None:
            self.modalities = [dict(m) for m in DEFAULT_MODALITIES]
        assert not self.tie_word_embeddings, "astropt3 has no lm_head to tie"
        assert self.tokeniser in ("affine", "aim", "jetformer"), f"unknown tokeniser {self.tokeniser!r}"
        names = [m["name"] for m in self.modalities]
        assert len(set(names)) == len(names), f"duplicate modality names: {names}"

    def modality_names(self) -> List[str]:
        """Alphabetical registry order — fixes sequence order everywhere."""
        return sorted(m["name"] for m in self.modalities)

    def modality(self, name: str) -> dict:
        return next(m for m in self.modalities if m["name"] == name)


@dataclass
class AstroPT3StreamingDatasetsArgs:
    """``astropt3_streaming`` dataset type.

    ``data_root`` points at a directory of pilot parquet shards written by
    ``astro/scripts/prepare_pilot_data.py`` (the train split directory), or
    the literal string ``"synthetic"`` for the offline synthetic stream used
    by smoke runs and gpu-marked tests.

    ``norm_stats`` optionally points at the data yaml holding the asinh
    p1/p99 calibration (``astro/configs/data/pilot_images_spectra.yaml``);
    without it the sequencer falls back to plain ``asinh(flux)`` (synthetic
    convention).

    NOTE: with DP > 1 the flattened per-modality tensors have different
    shapes on each DP rank, so ``general.ignore_sanity_checks`` must stay
    true (the DP input-difference sanity check all-gathers tensors and
    assumes equal shapes).
    """

    data_root: str
    is_astropt3_streaming: bool = True
    norm_stats: Optional[str] = None
    shuffle_buffer_size: int = 0
    # synthetic stream controls (data_root == "synthetic")
    synthetic_image_only_fraction: float = 0.3
    # append one object_id line per trained object to {path}.dp{rank} —
    # the no-replay audit trail for kill/resume verification
    object_id_log: Optional[str] = None
