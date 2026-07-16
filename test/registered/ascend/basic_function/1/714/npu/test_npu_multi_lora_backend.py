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

import multiprocessing as mp
import os
import unittest

from sglang.test.ascend.lora_utils import LoRAModelCase, LoRAAdaptor
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.ascend.lora_utils import (
    # ALL_OTHER_MULTI_LORA_MODELS,
    # CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
    run_lora_multiple_batch_on_model_cases,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=99, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=100, suite="stage-b-test-1-gpu-small-amd")

ALL_OTHER_MULTI_LORA_MODELS = [
    LoRAModelCase(
        base="meta-llama/Llama-3.1-8B-Instruct",
        adaptors=[
            LoRAAdaptor(
                name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
                prefill_tolerance=1e-1,
            ),
            LoRAAdaptor(
                name="nvidia/llama-3.1-nemoguard-8b-topic-control",
                prefill_tolerance=1e-1,
            ),
        ],
        max_loras_per_batch=2,
    ),
]

CI_MULTI_LORA_MODELS = [
    # multi-rank case
    LoRAModelCase(
        base="/home/weights/Qwen3.5-4B",
        adaptors=[
            LoRAAdaptor(
                name="/home/weights/qwen3.5-4b-mcat-lora",
                prefill_tolerance=1e-1,
                rouge_l_tolerance=0.9,
            ),
            LoRAAdaptor(
                name="/home/weights/qwen3.5-4b-neo4j-text2cypher-lora",
                prefill_tolerance=3e-1,
                rouge_l_tolerance=0.9,
            ),
        ],
        max_loras_per_batch=2,
        max_loaded_loras=4,
    ),
]


class TestMultiLoRABackend(CustomTestCase):
    def test_ci_lora_models_batch_splitting(self):
        run_lora_batch_splitting_equivalence_test(CI_MULTI_LORA_MODELS)

    def test_ci_lora_models_multi_batch(self):
        run_lora_multiple_batch_on_model_cases(CI_MULTI_LORA_MODELS)

    def test_all_lora_models(self):
        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_MULTI_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        run_lora_multiple_batch_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
