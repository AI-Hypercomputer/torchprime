# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from pydantic import BaseModel, model_validator
from enum import Enum
from typing import Optional


class QuantizationScheme(Enum):
    int4_weight_int8_dynamic_activation = "int4_weight_int8_dynamic_activation"


class QuantizationArgs(BaseModel):
    scheme: Optional[QuantizationScheme] = None
    group_size: Optional[int] = None
    spinquant: bool = False


class LoRAArgs(BaseModel):
    rank: int
    scale: float


class MoEArgs(BaseModel):
    num_experts: int = -1
    capacity_factor: float = 1.0  # capacity factor determines how many tokens each expert can choose
    auto_scale_F: bool = (  # noqa: N815
        True  # if true, rescales hidden_dim such that number of activated params is same as equivalent dense layer
    )
    use_shared_expert: bool = (
        True  # if true, creates a deterministic shared expert to be activated alongside the routed experts
    )
    use_token_choice: bool = False
    top_k: int = 1


class Size(BaseModel):
    height: int
    width: int


class VisionArgs(BaseModel):
    image_size: Size
    patch_size: Size

    # parameters for the encoder transformer
    dim: int
    n_layers: int
    n_heads: int
    mlp_ratio: float
    output_dim: int

    pixel_shuffle_ratio: float

    @model_validator(mode="before")
    @classmethod
    def preprocess_values(cls, data: dict) -> dict:
        data = data.copy()
        if "image_size" not in data:
            data["image_size"] = Size(height=data["image_height"], width=data["image_width"])
        if "patch_size" not in data:
            data["patch_size"] = Size(height=data["patch_height"], width=data["patch_width"])
        if "pixel_shuffle_ratio" not in data:
            data["pixel_shuffle_ratio"] = data["ps_ratio"]
        if "dim" not in data:
            data |= {
                "dim": 1408,
                "n_layers": 34,
                "n_heads": 16,
                "mlp_ratio": 4.0,
                "output_dim": 4096,
            }
        return data


class ModelArgs(BaseModel):
    dim: int = -1
    n_layers: int = -1
    n_heads: int = -1
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    ffn_exp: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False
    use_qk_norm: bool = False

    vision_args: Optional[VisionArgs] = None
    moe_args: Optional[MoEArgs] = None
    quantization_args: Optional[QuantizationArgs] = None
    lora_args: Optional[LoRAArgs] = None

    max_batch_size: int = 32
    max_seq_len: int = 2048

    @model_validator(mode="before")
    @classmethod
    def preprocess_values(cls, data: dict) -> dict:
        data = data.copy()
        if "model" in data:
            data = data["model"]

        data["n_kv_heads"] = data.get("n_kv_heads") or data.get("n_heads")
        data["moe_args"] = data.get("experts_choice_moe") or data.get("moe_args")

        if "vision_args" not in data:
            modalities = data.get("modalities", {})
            data["vision_args"] = modalities.get("image")

        return data

    @model_validator(mode="after")
    def validate(self) -> "ModelArgs":
        assert self.n_kv_heads <= self.n_heads, f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        assert self.dim % self.n_heads == 0, f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})"
        return self


def make_17b(max_batch_size, max_seq_len, tiny=False):
    payload = {
            "dim": 5120,
            "n_layers": 48,
            "n_heads": 40,
            "n_kv_heads": 8,
            "vocab_size": 202048,
            "multiple_of": 2048,
            "ffn_dim_multiplier": 1.2,
            "ffn_exp": 4.0,
            "norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "use_scaled_rope": True,
            "use_qk_norm": True,
            "vision_args": {
                "image_size": {
                "height": 336,
                "width": 336
                },
                "patch_size": {
                "height": 14,
                "width": 14
                },
                "dim": 1408,
                "n_layers": 34,
                "n_heads": 16,
                "mlp_ratio": 4.0,
                "output_dim": 4096,
                "pixel_shuffle_ratio": 0.5
            },
            "moe_args": {
                "num_experts": 16,
                "capacity_factor": 1.0,
                "auto_scale_F": True,
                "use_shared_expert": True,
                "use_token_choice": True,
                "top_k": 1
            },
            "quantization_args": None,
            "lora_args": None
    }

    args = ModelArgs(**payload, 
                     max_batch_size=max_batch_size,
                     max_seq_len=max_seq_len)

    if tiny:
        args.n_layers = 3
        args.dim = 1024
        args.n_heads = 8
        args.vision_args.n_layers = 1
    return args