# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Lopyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Qwen3 model configuration"""

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_rope_utils import rope_config_validation
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Qwen3Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Qwen3Model`]. It is used to instantiate a
    Qwen3 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of
    Qwen3-8B-beta [Qwen/Qwen3-8B-beta](https://huggingface.co/Qwen/Qwen3-8B-beta).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the Qwen3 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`Qwen3Model`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 22016):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_key_value_heads (`int`, *optional*, defaults to 32):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to `32`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 32768):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'llama3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'llama3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to use sliding window attention.
        sliding_window (`int`, *optional*, defaults to 4096):
            Sliding window attention (SWA) window size. If not specified, will default to `4096`.
        max_window_layers (`int`, *optional*, defaults to 28):
            The number of layers that use SWA (Sliding Window Attention). The bottom layers use SWA while the top use full attention.
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        use_qk_norm (`bool`, *optional*, defaults to `False`):
            Whether query and key in attention use norm
    ```python
    >>> from transformers import Qwen3Model, Qwen3Config

    >>> # Initializing a Qwen3 style configuration
    >>> configuration = Qwen3Config()

    >>> # Initializing a model from the Qwen3-8B style configuration
    >>> model = Qwen3Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "qwen3"
    keys_to_ignore_at_inference = ["past_key_values"]

    # Default tensor parallel plan for base model `Qwen3`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    _LAYER_GATE_TYPES = {
        "none",
        "shared_headwise",
        "shared_groupwise",
        "shared_elementwise",
        "basis",
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_bias=False,
        qkv_bias=False,
        attention_dropout=0.0,
        use_qk_norm=True,
        num_gate_groups=None,
        irg_attn_output_gate=False,
        irg_routing_hidden_size=None,
        hybrid_attn_output_gate=False,
        hybrid_gate_lambda=0.5,
        attn_output_gate_temperature=1.0,
        attn_output_gate_residual_alpha=0.0,
        independent_attn_output_gate=False,
        context_aware_attn_output_gate=False,
        elementwise_attn_output_gate=False,
        headwise_attn_output_gate=False,
        layer_gate_layout=None,
        basis_gate_enabled=False,
        basis_rank=None,
        basis_alpha_norm=False,
        basis_temperature=1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.qkv_bias = qkv_bias
        self.attention_dropout = attention_dropout
        self.use_qk_norm = use_qk_norm

        self.num_gate_groups = num_gate_groups
        self.irg_attn_output_gate = irg_attn_output_gate
        self.irg_routing_hidden_size = irg_routing_hidden_size
        self.hybrid_attn_output_gate = hybrid_attn_output_gate
        self.hybrid_gate_lambda = hybrid_gate_lambda
        self.attn_output_gate_temperature = attn_output_gate_temperature
        self.attn_output_gate_residual_alpha = attn_output_gate_residual_alpha
        self.independent_attn_output_gate = independent_attn_output_gate
        self.context_aware_attn_output_gate = context_aware_attn_output_gate
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.elementwise_attn_output_gate = elementwise_attn_output_gate
        self.layer_gate_layout = None
        self.basis_gate_enabled = basis_gate_enabled
        self.basis_rank = basis_rank
        self.basis_alpha_norm = basis_alpha_norm
        self.basis_temperature = basis_temperature

        if not 0.0 <= self.hybrid_gate_lambda <= 1.0:
            raise ValueError("`hybrid_gate_lambda` must be within [0, 1].")
        if self.attn_output_gate_temperature <= 0.0:
            raise ValueError("`attn_output_gate_temperature` must be > 0.")
        if not 0.0 <= self.attn_output_gate_residual_alpha <= 1.0:
            raise ValueError("`attn_output_gate_residual_alpha` must be within [0, 1].")
        if self.basis_temperature <= 0.0:
            raise ValueError("`basis_temperature` must be > 0.")
        if self.basis_rank is not None and self.basis_rank < 1:
            raise ValueError("`basis_rank` must be a positive integer.")

        if self.context_aware_attn_output_gate and not self.independent_attn_output_gate:
            raise ValueError("`context_aware_attn_output_gate=True` requires `independent_attn_output_gate=True`.")

        if self.headwise_attn_output_gate and self.elementwise_attn_output_gate:
            raise ValueError("`headwise_attn_output_gate` and `elementwise_attn_output_gate` cannot both be True.")

        legacy_num_gate_groups = None
        if self.headwise_attn_output_gate:
            legacy_num_gate_groups = 1
        elif self.elementwise_attn_output_gate:
            legacy_num_gate_groups = self.head_dim

        if self.num_gate_groups is not None:
            if self.num_gate_groups < 1:
                raise ValueError("`num_gate_groups` must be a positive integer.")
            if legacy_num_gate_groups is not None and self.num_gate_groups != legacy_num_gate_groups:
                raise ValueError(
                    "`num_gate_groups` cannot conflict with legacy gate flags "
                    "`headwise_attn_output_gate` / `elementwise_attn_output_gate`."
                )
        else:
            self.num_gate_groups = legacy_num_gate_groups

        if self.num_gate_groups is not None and self.head_dim % self.num_gate_groups != 0:
            raise ValueError(
                f"`head_dim` ({self.head_dim}) must be divisible by `num_gate_groups` ({self.num_gate_groups})."
            )

        if self.irg_attn_output_gate:
            if self.independent_attn_output_gate:
                raise ValueError("`irg_attn_output_gate=True` is only supported for shared gating.")
            if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
                raise ValueError(
                    "`irg_attn_output_gate=True` cannot be combined with legacy "
                    "`headwise_attn_output_gate` / `elementwise_attn_output_gate` flags."
                )
            if self.hybrid_attn_output_gate:
                raise ValueError("`irg_attn_output_gate=True` cannot be combined with `hybrid_attn_output_gate=True`.")
            if self.num_gate_groups is None or self.num_gate_groups <= 1:
                raise ValueError(
                    "`irg_attn_output_gate=True` requires `num_gate_groups` to be set to an integer > 1."
                )
            if self.attn_output_gate_temperature != 1.0 or self.attn_output_gate_residual_alpha != 0.0:
                raise ValueError(
                    "`irg_attn_output_gate=True` cannot be combined with temperature or residual gate tweaks."
                )
            if self.irg_routing_hidden_size is None:
                self.irg_routing_hidden_size = min(self.head_dim, 16)
            elif self.irg_routing_hidden_size < 1:
                raise ValueError("`irg_routing_hidden_size` must be a positive integer.")
        elif self.irg_routing_hidden_size is not None:
            raise ValueError("`irg_routing_hidden_size` requires `irg_attn_output_gate=True`.")

        if self.independent_attn_output_gate and not (
            self.headwise_attn_output_gate or self.elementwise_attn_output_gate
        ):
            raise ValueError(
                "`independent_attn_output_gate=True` requires either `headwise_attn_output_gate=True` "
                "or `elementwise_attn_output_gate=True`."
            )

        if self.independent_attn_output_gate and self.num_gate_groups not in (None, 1, self.head_dim):
            raise ValueError(
                "Independent gate currently supports only legacy headwise/elementwise modes. "
                "Set `num_gate_groups` via shared gating only."
            )

        if self.attn_output_gate_residual_alpha != 0.0 and self.num_gate_groups is None:
            raise ValueError(
                "`attn_output_gate_residual_alpha` requires attention output gating to be enabled."
            )

        if self.attn_output_gate_temperature != 1.0 and self.num_gate_groups is None:
            raise ValueError(
                "`attn_output_gate_temperature` requires attention output gating to be enabled."
            )

        if self.hybrid_attn_output_gate:
            if self.independent_attn_output_gate:
                raise ValueError("`hybrid_attn_output_gate=True` is only supported for shared gating.")
            if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
                raise ValueError(
                    "`hybrid_attn_output_gate=True` cannot be combined with legacy "
                    "`headwise_attn_output_gate` / `elementwise_attn_output_gate` flags."
                )
            if self.num_gate_groups is None or self.num_gate_groups <= 1:
                raise ValueError(
                    "`hybrid_attn_output_gate=True` requires `num_gate_groups` to be set to an integer > 1."
                )
            if self.attn_output_gate_temperature != 1.0 or self.attn_output_gate_residual_alpha != 0.0:
                raise ValueError(
                    "`hybrid_attn_output_gate=True` cannot be combined with temperature or residual gate tweaks."
                )

        if layer_gate_layout is not None:
            if any(
                (
                    self.num_gate_groups is not None,
                    self.irg_attn_output_gate,
                    self.hybrid_attn_output_gate,
                    self.independent_attn_output_gate,
                    self.context_aware_attn_output_gate,
                    self.headwise_attn_output_gate,
                    self.elementwise_attn_output_gate,
                    self.attn_output_gate_temperature != 1.0,
                    self.attn_output_gate_residual_alpha != 0.0,
                )
            ):
                raise ValueError(
                    "`layer_gate_layout` cannot be combined with legacy global gate configuration flags."
                )
            self.layer_gate_layout = self._normalize_layer_gate_layout(
                layer_gate_layout=layer_gate_layout,
                num_hidden_layers=self.num_hidden_layers,
                head_dim=self.head_dim,
            )
            self.basis_gate_enabled = any(
                layer_spec["gate_type"] == "basis" for layer_spec in self.layer_gate_layout
            )
            if self.basis_gate_enabled:
                basis_ranks = sorted(
                    {layer_spec["basis_rank"] for layer_spec in self.layer_gate_layout if layer_spec["gate_type"] == "basis"}
                )
                self.basis_rank = basis_ranks[0] if len(basis_ranks) == 1 else None
                basis_alpha_norms = {
                    layer_spec["basis_alpha_norm"]
                    for layer_spec in self.layer_gate_layout
                    if layer_spec["gate_type"] == "basis"
                }
                self.basis_alpha_norm = len(basis_alpha_norms) == 1 and next(iter(basis_alpha_norms))
                basis_temperatures = {
                    layer_spec["basis_temperature"]
                    for layer_spec in self.layer_gate_layout
                    if layer_spec["gate_type"] == "basis"
                }
                self.basis_temperature = next(iter(basis_temperatures)) if len(basis_temperatures) == 1 else None
        elif self.basis_gate_enabled and self.basis_rank is None:
            raise ValueError("`basis_gate_enabled=True` requires `basis_rank` to be set.")

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @classmethod
    def _normalize_layer_gate_layout(cls, layer_gate_layout, num_hidden_layers, head_dim):
        if len(layer_gate_layout) != num_hidden_layers:
            raise ValueError(
                f"`layer_gate_layout` must contain exactly {num_hidden_layers} layer specs, "
                f"got {len(layer_gate_layout)}."
            )

        normalized_layout = []
        for layer_index, raw_spec in enumerate(layer_gate_layout):
            if not isinstance(raw_spec, dict):
                raise ValueError(f"Layer gate spec at index {layer_index} must be a dict.")

            spec = copy.deepcopy(raw_spec)
            gate_type = spec.get("gate_type")
            if gate_type not in cls._LAYER_GATE_TYPES:
                raise ValueError(
                    f"Layer gate spec at index {layer_index} has invalid gate_type {gate_type!r}. "
                    f"Supported values: {sorted(cls._LAYER_GATE_TYPES)}."
                )

            normalized_spec = {
                "gate_type": gate_type,
                "num_gate_groups": spec.get("num_gate_groups"),
                "basis_rank": spec.get("basis_rank"),
                "basis_alpha_norm": bool(spec.get("basis_alpha_norm", False)),
                "basis_temperature": float(spec.get("basis_temperature", 1.0)),
            }

            if normalized_spec["basis_temperature"] <= 0.0:
                raise ValueError(
                    f"Layer gate spec at index {layer_index} must have `basis_temperature > 0`."
                )

            if gate_type == "none":
                normalized_spec["num_gate_groups"] = None
                normalized_spec["basis_rank"] = None
            elif gate_type == "shared_headwise":
                normalized_spec["num_gate_groups"] = 1
                normalized_spec["basis_rank"] = None
            elif gate_type == "shared_elementwise":
                normalized_spec["num_gate_groups"] = head_dim
                normalized_spec["basis_rank"] = None
            elif gate_type == "shared_groupwise":
                num_gate_groups = normalized_spec["num_gate_groups"]
                if not isinstance(num_gate_groups, int) or num_gate_groups < 1:
                    raise ValueError(
                        f"Layer gate spec at index {layer_index} requires a positive integer `num_gate_groups`."
                    )
                if head_dim % num_gate_groups != 0:
                    raise ValueError(
                        f"Layer gate spec at index {layer_index} has `num_gate_groups={num_gate_groups}`, "
                        f"which does not divide `head_dim={head_dim}`."
                    )
                normalized_spec["basis_rank"] = None
            elif gate_type == "basis":
                basis_rank = normalized_spec["basis_rank"]
                if not isinstance(basis_rank, int) or basis_rank < 1:
                    raise ValueError(
                        f"Layer gate spec at index {layer_index} requires a positive integer `basis_rank`."
                    )
                normalized_spec["num_gate_groups"] = None

            normalized_layout.append(normalized_spec)

        return normalized_layout

    def uses_layer_gate_layout(self):
        return self.layer_gate_layout is not None

    def get_layer_gate_spec(self, layer_idx):
        if self.layer_gate_layout is None:
            return None
        if layer_idx is None or layer_idx < 0 or layer_idx >= len(self.layer_gate_layout):
            raise IndexError(f"Layer index {layer_idx} is out of range for `layer_gate_layout`.")
        return copy.deepcopy(self.layer_gate_layout[layer_idx])

