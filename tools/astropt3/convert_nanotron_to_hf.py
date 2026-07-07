"""Convert an AstroPT3 nanotron checkpoint to the HF release format.

Command (in the [train] env — needs nanotron, astropt3 and a GPU):
    torchrun --nproc_per_node=1 tools/astropt3/convert_nanotron_to_hf.py \
        --checkpoint_path=checkpoints/10 --save_path=astropt3-hf

The saved model reloads with::

    import astropt3  # registers the Auto classes
    model = AutoModel.from_pretrained(save_path)
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import torch
from nanotron.config import AstroPT3Config as NanotronAstroPT3Config
from nanotron.models import init_on_device_and_dtype
from nanotron.models.astropt3 import AstroPT3ForTraining

sys.path.append(str(Path(__file__).parent))
from convert_weights import get_config_mapping, get_weight_mapping, load_nanotron_model  # noqa: E402

from astropt3 import AstroPT3Config as HFAstroPT3Config  # noqa: E402
from astropt3 import AstroPT3Model  # noqa: E402


def _handle_attention_block(
    qkv: torch.Tensor,
    part: Literal["q", "k", "v"],
    n_q_heads: int,
    n_kv_heads: int,
    d_qk: int,
    interleave: bool,
) -> torch.Tensor:
    # See examples/smollm3: select the proper chunk of nanotron's fused qkv.
    # rope_interleaved is False for astropt3, so no permutation by default.
    def interleave_weight(w: torch.Tensor):
        w_new = []
        for head_w in w.split(d_qk):
            head_w = head_w.view(d_qk // 2, 2, -1).transpose(0, 1).reshape(d_qk, -1)
            w_new.append(head_w)
        return torch.cat(w_new)

    assert part in ["q", "k", "v"], "part must be one of [q, k, v]"
    index_end_q = n_q_heads * d_qk
    index_end_k = index_end_q + n_kv_heads * d_qk
    if part == "q":
        return interleave_weight(qkv[:index_end_q]) if interleave else qkv[:index_end_q]
    if part == "k":
        return interleave_weight(qkv[index_end_q:index_end_k]) if interleave else qkv[index_end_q:index_end_k]
    return qkv[index_end_k:]


def _handle_gate_up_proj(gate_up_proj: torch.Tensor, gate: bool) -> torch.Tensor:
    weight_size = gate_up_proj.shape[0] // 2
    return gate_up_proj[:weight_size] if gate else gate_up_proj[weight_size:]


def convert_nt_to_hf(
    nanotron_model: AstroPT3ForTraining,
    hf_model: AstroPT3Model,
    model_config: NanotronAstroPT3Config,
    interleave_qkv: bool = False,
):
    """Copy nanotron weights into the HF model in-place."""
    nanotron_model_state_dict = nanotron_model.state_dict()
    hf_to_nt = get_weight_mapping(model_config, nt_to_hf=False)

    for module_name_hf, module_hf in hf_model.named_modules():
        for param_name_hf, param_hf in module_hf.named_parameters(recurse=False):
            hf_key = f"{module_name_hf}.{param_name_hf}"
            nanotron_key = hf_to_nt[hf_key]
            param = nanotron_model_state_dict[nanotron_key]

            if "qkv_proj" in nanotron_key:
                proj_name = module_name_hf.split(".")[-1][0]  # q/k/v from {q,k,v}_proj
                param = _handle_attention_block(
                    param,
                    proj_name,
                    model_config.num_attention_heads,
                    model_config.num_key_value_heads,
                    model_config.hidden_size // model_config.num_attention_heads,
                    interleave_qkv,
                )
            elif "gate_up_proj" in nanotron_key:
                param = _handle_gate_up_proj(param, gate="gate" in module_name_hf)

            with torch.no_grad():
                param_hf.copy_(param)


def get_hf_config(config: NanotronAstroPT3Config) -> HFAstroPT3Config:
    attrs = {key: getattr(config, value) for key, value in get_config_mapping(nt_to_hf=False).items()}
    return HFAstroPT3Config(**attrs)


def convert_checkpoint_and_save(checkpoint_path: Path, save_path: Path):
    """nanotron checkpoint dir -> HF ``save_pretrained`` dir."""
    with open(checkpoint_path / "model_config.json") as f:
        model_config = NanotronAstroPT3Config(**json.load(f))
    nanotron_model = load_nanotron_model(model_config=model_config, checkpoint_path=checkpoint_path)

    with init_on_device_and_dtype(torch.device("cuda"), torch.bfloat16):
        hf_model = AstroPT3Model._from_config(get_hf_config(model_config))

    convert_nt_to_hf(nanotron_model, hf_model, model_config)
    hf_model.save_pretrained(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert AstroPT3 nanotron weights to HF format")
    parser.add_argument("--checkpoint_path", type=Path, required=True, help="nanotron checkpoint dir")
    parser.add_argument("--save_path", type=Path, required=True, help="output HF model dir")
    args = parser.parse_args()
    convert_checkpoint_and_save(checkpoint_path=args.checkpoint_path, save_path=args.save_path)
