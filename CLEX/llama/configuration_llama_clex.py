# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" LLaMA model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers import LlamaConfig


logger = logging.get_logger(__name__)

LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class CLEXLlamaConfig(LlamaConfig):

    model_type = "llama"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        rope_scaling=None,
        use_flashattn=True,
        log_scale=True,
        pretraining_tp=1,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.pretraining_tp = pretraining_tp
        self.use_flashattn = use_flashattn
        self.log_scale = log_scale
        # self.rope_theta = 10000
        # self.max_position_embeddings = 4096
        # self.data_length = 4096
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()


    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        # if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
        #     raise ValueError(
        #         "`rope_scaling` must be a dictionary with with two fields, `name` and `factor`, "
        #         f"got {self.rope_scaling}"
        #     )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_max_factor = self.rope_scaling.get("max_factor", None)
        rope_scaling_param_factor = self.rope_scaling.get("param_factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic", "clex"]:
            raise ValueError(
                f"`rope_scaling`'s name field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        # if rope_scaling_max_factor is None or not isinstance(rope_scaling_max_factor, float) or rope_scaling_max_factor <= 1.0:
        #     raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_max_factor}")
        # if rope_scaling_param_factor is None or not isinstance(rope_scaling_param_factor, float) or rope_scaling_param_factor <= 1.0:
        #     raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_param_factor}")
