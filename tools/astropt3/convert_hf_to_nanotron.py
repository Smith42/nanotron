"""Convert an AstroPT3 HF checkpoint to nanotron format.

Command (in the [train] env — needs nanotron, astropt3 and a GPU):
    torchrun --nproc_per_node=1 tools/astropt3/convert_hf_to_nanotron.py \
        --checkpoint_path=astropt3-hf --save_path=nanotron-checkpoint
"""

import dataclasses
import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import nanotron
import torch
from nanotron.config import AstroPT3Config as NanotronAstroPT3Config
from nanotron.models.astropt3 import AstroPT3ForTraining

sys.path.append(str(Path(__file__).parent))
from convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model  # noqa: E402

from astropt3 import AstroPT3Config as HFAstroPT3Config  # noqa: E402
from astropt3 import AstroPT3Model  # noqa: E402


def _handle_attention_block(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    n_q_heads: int,
    n_kv_heads: int,
    d_qk: int,
    interleave: bool,
) -> torch.Tensor:
    # Inverse of the nt->hf split (see examples/llama); rope_interleaved is
    # False for astropt3 so no permutation by default.
    def interleave_weight(w: torch.Tensor):
        w_new = []
        for head_w in w.split(d_qk):
            head_w = head_w.view(2, d_qk // 2, -1).transpose(0, 1).reshape(d_qk, -1)
            w_new.append(head_w)
        return torch.cat(w_new)

    q = interleave_weight(q) if interleave else q
    k = interleave_weight(k) if interleave else k
    return torch.cat([q, k, v])


def convert_hf_to_nt(
    hf_model: AstroPT3Model,
    nanotron_model: AstroPT3ForTraining,
    config: NanotronAstroPT3Config,
    interleave_qkv: bool = False,
):
    """Copy HF weights into the nanotron model in-place."""
    hf_sd = hf_model.state_dict()
    nt_to_hf = get_weight_mapping(config, nt_to_hf=True)

    for module_name_nt, module_nt in nanotron_model.named_modules():
        for param_name_nt, param_nt in module_nt.named_parameters(recurse=False):
            nt_key = f"{module_name_nt}.{param_name_nt}"
            if nt_key not in nt_to_hf:
                raise KeyError(f"no HF mapping for nanotron parameter {nt_key}")
            if "qkv_proj" in nt_key:
                key_k, key_q, key_v = sorted(nt_to_hf[nt_key])
                param = _handle_attention_block(
                    hf_sd[key_q],
                    hf_sd[key_k],
                    hf_sd[key_v],
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.hidden_size // config.num_attention_heads,
                    interleave_qkv,
                )
            elif "gate_up_proj" in nt_key:
                key_gate, key_up = sorted(nt_to_hf[nt_key])
                param = torch.cat([hf_sd[key_gate], hf_sd[key_up]])
            else:
                param = hf_sd[nt_to_hf[nt_key]]

            with torch.no_grad():
                param_nt.copy_(param)


def get_nanotron_config(config: HFAstroPT3Config) -> NanotronAstroPT3Config:
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=True).items()}
    # training-machine attention path (the non-packed path is not maintained)
    attrs["_attn_implementation"] = "flash_attention_2"
    attrs["_use_qkv_packed"] = True
    attrs["_use_doc_masking"] = True
    return NanotronAstroPT3Config(**attrs)


def convert_checkpoint_and_save(checkpoint_path: Path, save_path: Path):
    """HF ``save_pretrained`` dir -> nanotron checkpoint dir."""
    hf_model = AstroPT3Model.from_pretrained(checkpoint_path).cuda().bfloat16()
    model_config = get_nanotron_config(hf_model.config)
    nanotron_model = load_nanotron_model(model_config=model_config)

    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=1, pipeline_parallel_size=1, tensor_parallel_size=1
    )
    convert_hf_to_nt(hf_model, nanotron_model, model_config)
    nanotron.serialize.save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=save_path)
    with open(save_path / "model_config.json", "w+") as f:
        json.dump(dataclasses.asdict(model_config), f)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert AstroPT3 HF weights to nanotron format")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="HF model dir")
    parser.add_argument("--save_path", type=Path, required=True, help="output nanotron checkpoint dir")
    args = parser.parse_args()
    convert_checkpoint_and_save(checkpoint_path=args.checkpoint_path, save_path=args.save_path)
