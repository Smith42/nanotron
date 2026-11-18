"""AstroPT3 config: a Qwen2/SmolLM3 body driven by continuous-modality inputs.

Two additions over Qwen2Config:

- ``modalities``: list of per-modality dicts (name/input_size/patch_size,
  position contract, family/source/record keys, image band or format, token ids,
  legacy loss weight)
  mirroring the HF-side
  ``astropt3.configuration_astropt3.DEFAULT_MODALITIES``. Registry order is
  alphabetical by name everywhere.
- ``AstroPT3StreamingDatasetsArgs``: the ``astropt3_streaming`` dataset type
  consumed by ``run_train.py`` (packed multimodal micro-batches built by the
  ``astropt3`` package's ``data/nanotron_loader.py``).

Named modalities own fixed three-id blocks. Configs must size ``vocab_size``
to cover their active token ids; there is no text vocabulary and no lm_head.
"""

from dataclasses import dataclass
from typing import List, Optional

from nanotron.config.models_config import Qwen2Config

# Keep the fixed and reserved blocks in sync with astropt3/tokenization.py.
_MODALITY_ID_BLOCKS = {
    "spectra": 5,
    "Z": 8,
    "ebv": 11,
    "photometry": 14,
    "galaxies": 23,
    "legacy_north_g": 160,
    "legacy_north_r": 163,
    "legacy_north_z": 166,
    "legacy_south_g": 169,
    "legacy_south_r": 172,
    "legacy_south_i": 175,
    "legacy_south_z": 178,
    "hsc_g": 181,
    "hsc_r": 184,
    "hsc_i": 187,
    "hsc_z": 190,
    "hsc_y": 193,
}
_FIXED_TOKEN_IDS = {token_id for base in _MODALITY_ID_BLOCKS.values() for token_id in range(base, base + 3)}
_RESERVED_TOKEN_IDS = {0, 1, *range(2, 5), *range(20, 23), *range(53, 56)}

# Keep in sync with astropt3/configuration_astropt3.py.
DEFAULT_MODALITIES = [
    {
        "name": name,
        "input_size": 64,
        "patch_size": 8,
        "pos_type": "index",
        "pos_input_size": 1,
        "max_positions": 144,
        "family": "image",
        "source": "legacy_north",
        "record_keys": ["image"],
        "band": f"des-{name.removeprefix('legacy_north_')}",
        "token_ids": [base, base + 1, base + 2],
    }
    for name, base in _MODALITY_ID_BLOCKS.items()
    if name.startswith("legacy_north_")
] + [
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
    },
]


def _validate_image_modality(modality: dict) -> None:
    name = modality["name"]
    image_format = modality.get("image_format", "physical")
    if image_format == "rgb":
        if modality.get("band") is not None:
            raise ValueError(f"modality {name!r} RGB images do not support band")
        channels = 3
    elif image_format == "physical":
        band = modality.get("band")
        if not isinstance(band, str) or not band.strip() or band.startswith("rgb-"):
            raise ValueError(f"modality {name!r} physical images must declare a calibrated band")
        channels = 1
    else:
        raise ValueError(f"modality {name!r} has invalid image_format {image_format!r} (physical | rgb)")
    keys = modality["record_keys"]
    if not isinstance(keys, (list, tuple)) or len(keys) != 1 or not isinstance(keys[0], str) or not keys[0].strip():
        raise ValueError(f"modality {name!r} image record_keys must contain exactly one nonempty key")
    patch_size = modality.get("patch_size")
    if type(patch_size) is not int or patch_size < 1:
        raise ValueError(f"modality {name!r} patch_size must be a positive integer")
    expected = channels * patch_size**2
    if modality["input_size"] != expected:
        raise ValueError(
            f"modality {name!r} input_size must be {expected} for {image_format} images with patch_size={patch_size}"
        )


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
    # arcsinh knee (nMgy) of the physical spectra normalization (ADR 0007),
    # the spectra counterpart of image_norm_divisor; threaded and converted
    # the same way (mirrors the HF-side default)
    spectra_norm_divisor: float = 10.0
    # center-outward spiral image patch order (ADR 0004); threaded into the
    # sequencer like image_norm_divisor and carried into converted HF
    # checkpoints (mirrors the HF-side default).
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
        for field in ("jetformer_flow_steps", "jetformer_flow_hidden", "jetformer_flow_depth", "jetformer_flow_heads"):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.jetformer_flow_hidden % self.jetformer_flow_heads:
            raise ValueError("jetformer_flow_hidden must be divisible by jetformer_flow_heads")
        raw_modalities = DEFAULT_MODALITIES if self.modalities is None else self.modalities
        defaults_by_name = {modality["name"]: modality for modality in DEFAULT_MODALITIES}
        completed = []
        used_token_ids = _RESERVED_TOKEN_IDS.copy()
        for raw in raw_modalities:
            modality = dict(raw)
            name = modality.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("modality name must be a nonempty string")
            for key in ("channel_tokenization", "band_order"):
                if key in modality:
                    raise ValueError(f"modality {name!r} does not support {key}")
            defaults = defaults_by_name.get(name, {})
            for key in ("family", "source", "record_keys", "band"):
                if key not in modality and key in defaults:
                    modality[key] = defaults[key]
            base = _MODALITY_ID_BLOCKS.get(name)
            fixed_ids = tuple(range(base, base + 3)) if base is not None else None
            if fixed_ids is not None:
                modality.setdefault("token_ids", fixed_ids)
            missing = [key for key in ("family", "source", "record_keys", "token_ids") if key not in modality]
            if missing:
                raise ValueError(f"modality {name!r} is missing {', '.join(missing)}")
            try:
                token_ids = tuple(int(token_id) for token_id in modality["token_ids"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"modality {name!r} has invalid token_ids") from error
            if len(token_ids) != 3 or token_ids != tuple(range(token_ids[0], token_ids[0] + 3)):
                raise ValueError(f"modality {name!r} token_ids must be three consecutive ids")
            if fixed_ids is not None and token_ids != fixed_ids:
                raise ValueError(f"modality {name!r} must use token_ids {fixed_ids}")
            blocked_ids = used_token_ids if fixed_ids is not None else used_token_ids | _FIXED_TOKEN_IDS
            if overlap := blocked_ids.intersection(token_ids):
                raise ValueError(f"modality {name!r} token_ids collide at {sorted(overlap)}")
            modality["token_ids"] = list(token_ids)
            family = modality["family"]
            if family not in ("image", "spectrum", "scalar"):
                raise ValueError(f"modality {name!r} has invalid family {family!r}")
            dim = modality.get("input_size")
            if type(dim) is not int or dim < 1:
                raise ValueError(f"modality {name!r} input_size must be a positive integer")
            if family == "image":
                _validate_image_modality(modality)
            elif modality.get("band") is not None or modality.get("image_format", "physical") != "physical":
                raise ValueError(f"modality {name!r} band and image_format are only supported for images")
            if family != "scalar":
                if dim % 2:
                    raise ValueError(f"modality {modality['name']!r} flow requires an even input_size, got {dim}")
                pos_type = modality.get("pos_type", "index")
                if pos_type not in ("index", "continuous"):
                    raise ValueError(f"unknown pos_type {pos_type!r}")
                for field, default in (("max_positions", 1024), ("pos_input_size", 1)):
                    value = modality.get(field, default)
                    if type(value) is not int or value < 1:
                        raise ValueError(f"modality {modality['name']!r} {field} must be a positive integer")
            used_token_ids.update(token_ids)
            completed.append(modality)
        self.modalities = completed
        required_vocab = max(64, max(used_token_ids) + 1)
        if self.vocab_size < required_vocab:
            raise ValueError(f"vocab_size={self.vocab_size} cannot hold modality token id {required_vocab - 1}")
        if self.tie_word_embeddings:
            raise ValueError("astropt3 has no lm_head to tie")
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
    """Explicit YAML-configured LSDB catalog streams.

    ``catalog_stream`` is required. Validation/decoding lives in ``astropt3``.
    DataLoader workers provide concurrency, without a resumable stream cursor.
    """

    is_astropt3_streaming: bool = True
    # Total optimizer-step exposures per batch from each loaded draw.
    echo_factor: int = 1
    catalog_stream: Optional[dict] = None

    def __post_init__(self):
        if type(self.echo_factor) is not int or self.echo_factor < 1:
            raise ValueError("echo_factor must be a positive integer")
        if not isinstance(self.catalog_stream, dict) or not self.catalog_stream.get("catalogs"):
            raise ValueError("catalog_stream.catalogs is required; migrate legacy dataset YAML")
