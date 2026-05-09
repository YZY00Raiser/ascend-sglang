# Copyright 2023-2024 SGLang Team
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
# ==============================================================================
"""
NPU LoRA Tensor Parallel Test

This test verifies LoRA functionality with tensor parallelism on NPU.
Test cases are ported from GPU test_lora_tp.py for NPU compatibility.

[Test Category] Integration
[Test Target] LoRA with TP (Tensor Parallel) on NPU
"""

import multiprocessing as mp
import os
import unittest
from typing import List, Optional

import torch

from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.lora_utils import LoRAModelCase, run_lora_test_one_by_one
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_npu_ci(
    est_time=300,
    suite="nightly-2-npu-a3",
    nightly=True,
)

# NPU compatible prompts
DEFAULT_PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
]

# NPU supported torch dtypes
TORCH_DTYPES = [torch.float16]

# NPU compatible LoRA model cases
NPU_LORA_MODELS = [
    LoRAModelCase(
        base=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
        loras=[],
        max_lora_rank=32,
    ),
]

# Multi-LoRA test cases for NPU
NPU_MULTI_LORA_MODELS = [
    LoRAModelCase(
        base=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
        loras=[],
        max_lora_rank=32,
    ),
]


class TestNpuLoRATP(CustomTestCase):
    """Test LoRA with Tensor Parallel on NPU."""

    def _run_tp_on_model_cases(
        self,
        model_cases: List[LoRAModelCase],
        enable_lora_overlap_loading: Optional[bool] = None,
    ):
        """Run TP tests on model cases."""
        tp_list = [2]  # TP=2 for NPU multi-card testing
        for model_case in model_cases:
            prompts = DEFAULT_PROMPTS
            for tp_size in tp_list:
                model_case.tp_size = tp_size
                for torch_dtype in TORCH_DTYPES:
                    run_lora_test_one_by_one(
                        prompts,
                        model_case,
                        torch_dtype,
                        max_new_tokens=32,
                        enable_lora_overlap_loading=enable_lora_overlap_loading,
                        test_tag=f"tp={tp_size}, enable_lora_overlap_loading={enable_lora_overlap_loading}",
                        attention_backend="ascend",
                    )

    def test_npu_lora_tp_basic(self):
        """Test basic LoRA functionality with TP=2 on NPU."""
        self._run_tp_on_model_cases(NPU_LORA_MODELS)

    def test_npu_lora_tp_overlap_loading(self):
        """Test LoRA with overlap loading and TP=2 on NPU."""
        self._run_tp_on_model_cases(
            NPU_MULTI_LORA_MODELS, enable_lora_overlap_loading=True
        )

    def test_npu_lora_tp_llama3_1_8b(self):
        """Test LoRA TP with Llama-3.1-8B on NPU."""
        if is_in_ci():
            self.skipTest("Skipping large model test in CI")

        model_cases = [
            LoRAModelCase(
                base=LLAMA_3_1_8B_INSTRUCT_WEIGHTS_PATH,
                loras=[],
                max_lora_rank=32,
            ),
        ]
        self._run_tp_on_model_cases(model_cases)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
