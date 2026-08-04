"""AstroPT3 config: a Qwen2/SmolLM3 body driven by continuous-modality inputs.

Two additions over Qwen2Config:

- ``modalities``: list of per-modality dicts (name/input_size/patch_size,
  position contract, family/source/record keys, token ids, legacy loss weight)
  mirroring the HF-side
  ``astropt3.configuration_astropt3.DEFAULT_MODALITIES``. Registry order is
  alphabetical by name everywhere.
- ``AstroPT3StreamingDatasetsArgs``: the ``astropt3_streaming`` dataset type
  consumed by ``run_train.py`` (packed multimodal micro-batches built by the
  ``astropt3`` package's ``data/nanotron_loader.py``).

Existing token ids 0–16 are frozen. ADR 0013 configs consume reserved blocks
through id 63 before explicitly enlarging ``vocab_size``; there is no text
vocabulary and no lm_head.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from nanotron.config.models_config import Qwen2Config

# Pinned to the verified MMU pilot schemas (images (3,152,152) patch 8;
# DESI spectra 7781 bins patch 256; ADR 0008 one-token scalar spans with
# GMM heads under both tokenisers). Must stay in sync with the HF-side
# DEFAULT_MODALITIES in astropt3/configuration_astropt3.py.
DEFAULT_MODALITIES = [
    {
        "name": "images",
        "input_size": 192,
        "patch_size": 8,
        "pos_type": "index",
        "pos_input_size": 1,
        "max_positions": 361,
        "family": "image",
        "source": "legacy",
        "record_keys": ["image"],
        "token_ids": [2, 3, 4],
        "loss_weight": 1.0,
    },
    {
        "name": "spectra",
        "input_size": 256,
        "patch_size": 256,
        "pos_type": "continuous",
        "pos_input_size": 1,
        "max_positions": 1024,
        "family": "spectrum",
        "source": "desi",
        "record_keys": ["spectrum"],
        "token_ids": [5, 6, 7],
        "loss_weight": 1.0,
    },
    {
        "name": "Z",
        "input_size": 1,
        "patch_size": 1,
        "pos_type": "index",
        "pos_input_size": 1,
        "max_positions": 1,
        "family": "scalar",
        "source": "desi",
        "record_keys": ["Z"],
        "token_ids": [8, 9, 10],
        "loss_weight": 0.1,
        "scalar": True,
    },
    {
        "name": "ebv",
        "input_size": 1,
        "patch_size": 1,
        "pos_type": "index",
        "pos_input_size": 1,
        "max_positions": 1,
        "family": "scalar",
        "source": "legacy",
        "record_keys": ["ebv"],
        "token_ids": [11, 12, 13],
        "loss_weight": 0.1,
        "scalar": True,
    },
    {
        "name": "photometry",
        "input_size": 3,
        "patch_size": 1,
        "pos_type": "index",
        "pos_input_size": 1,
        "max_positions": 1,
        "family": "scalar",
        "source": "legacy",
        "record_keys": ["flux_g", "flux_r", "flux_z"],
        "token_ids": [14, 15, 16],
        "loss_weight": 0.1,
        "scalar": True,
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
    tokeniser: str = field(default="affine")  # affine, aim, or jetformer
    huber_delta: float = 1.0
    loss_aggregation: str = "legacy_modality_mean"
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
    # ADR 0008 scalar modalities: mixture count of the scalar GMM heads
    # (used under BOTH tokenisers; carried into converted HF checkpoints)
    scalar_gmm_k: int = 5
    # arcsinh divisor (nMgy) of the physical image normalization; threaded
    # into the sequencer by astro's build_astropt3_dataloader and carried
    # into converted HF checkpoints (mirrors the HF-side default)
    image_norm_divisor: float = 0.01
    # arcsinh knee (nMgy) of the physical spectra normalization (ADR 0007),
    # the spectra counterpart of image_norm_divisor; threaded and converted
    # the same way (mirrors the HF-side default)
    spectra_norm_divisor: float = 10.0
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
        raw_modalities = (
            [dict(modality) for modality in DEFAULT_MODALITIES] if self.modalities is None else self.modalities
        )
        legacy = {modality["name"]: modality for modality in DEFAULT_MODALITIES}
        completed = []
        used_token_ids = {0, 1}
        for raw in raw_modalities:
            modality = dict(raw)
            defaults = legacy.get(modality.get("name"), {})
            for key in ("family", "source", "record_keys", "token_ids"):
                if key not in modality and key in defaults:
                    modality[key] = defaults[key]
            missing = [key for key in ("family", "source", "record_keys", "token_ids") if key not in modality]
            if missing:
                raise ValueError(f"modality {modality.get('name')!r} is missing {', '.join(missing)}")
            try:
                token_ids = tuple(int(token_id) for token_id in modality["token_ids"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"modality {modality['name']!r} has invalid token_ids"
                ) from error
            if len(token_ids) != 3 or token_ids != tuple(
                range(token_ids[0], token_ids[0] + 3)
            ):
                raise ValueError(
                    f"modality {modality['name']!r} token_ids must be three consecutive ids"
                )
            modality["token_ids"] = list(token_ids)
            overlap = used_token_ids.intersection(token_ids)
            if overlap:
                raise ValueError(f"modality {modality['name']!r} token_ids collide at {sorted(overlap)}")
            if modality["family"] not in ("image", "spectrum", "scalar"):
                raise ValueError(f"modality {modality['name']!r} has invalid family {modality['family']!r}")
            modality["scalar"] = modality["family"] == "scalar"
            used_token_ids.update(token_ids)
            completed.append(modality)
        self.modalities = completed
        required_vocab = max(64, max(used_token_ids) + 1)
        if self.vocab_size < required_vocab:
            raise ValueError(f"vocab_size={self.vocab_size} cannot hold modality token id {required_vocab - 1}")
        if self.tie_word_embeddings:
            raise ValueError("astropt3 has no lm_head to tie")
        if self.tokeniser not in ("affine", "aim", "jetformer"):
            raise ValueError(f"unknown tokeniser {self.tokeniser!r}")
        if self.loss_aggregation not in ("legacy_modality_mean", "family"):
            raise ValueError(f"unknown loss_aggregation {self.loss_aggregation!r}")
        names = [modality["name"] for modality in completed]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate modality names: {names}")

    def modality_names(self) -> List[str]:
        """Alphabetical registry order — fixes sequence order everywhere."""
        return sorted(modality["name"] for modality in self.modalities or ())

    def modality(self, name: str) -> dict:
        return next(modality for modality in self.modalities or () if modality["name"] == name)


@dataclass
class AstroPT3StreamingDatasetsArgs:
    """``astropt3_streaming`` dataset type.

    ``data_root`` is ``"mmu"`` — the MMU HATS catalogs streamed live from the
    HF hub (ADR 0006; the local parquet reshard and its prep script are
    gone) — or the literal string ``"synthetic"`` for the offline synthetic
    stream used by smoke runs and gpu-marked tests. Any other value raises
    in the loader, so a config still naming the retired corpus fails loudly.

    ``match_index`` is the precomputed crossmatch parquet built offline by
    ``astro/scripts/build_match_index.py`` (ADR 0006). It is **mandatory** for
    ``data_root: mmu`` and DEFINES the corpus (ADR 0011 as amended
    2026-08-04): one pass over its LegacySurvey cells emits matched pairs,
    unmatched images, and the globally unmatched spectra of the cells it owns.
    There is no standalone source, no weighting, and no degrade-to-images
    fallback — the loader raises without an index. ``$ASTROPT3_MATCH_INDEX``
    is the fallback when the field is unset.

    ``num_loading_workers`` is capped by the corpus, not the machine: the
    published index holds 173 cells (165 train), partitions are dealt to DP
    ranks, so a rank owns ``floor(165 / dp)`` of them and the loader raises
    if it has more workers than that. ``datasets`` would only warn and
    silently stop the surplus.

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
    match_index: Optional[str] = None
    norm_stats: Optional[str] = None
    # synthetic stream controls (data_root == "synthetic")
    synthetic_image_only_fraction: float = 0.3
    synthetic_spectrum_only_fraction: float = 0.0
    # append one object_id line per trained object to {path}.dp{rank} —
    # the no-replay audit trail for kill/resume verification
    object_id_log: Optional[str] = None
