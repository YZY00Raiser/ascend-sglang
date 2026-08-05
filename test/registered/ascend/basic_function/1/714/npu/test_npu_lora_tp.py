import multiprocessing as mp
import os
import unittest
from typing import List, Optional

import torch

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.lora_utils import (
    # ALL_OTHER_LORA_MODELS,
    # CI_LORA_MODELS,
    # CI_MULTI_LORA_MODELS,
    DEFAULT_PROMPTS,
    # TORCH_DTYPES,
    LoRAModelCase,
    run_lora_test_one_by_one, LoRAAdaptor,
)
from sglang.test.test_utils import CustomTestCase
TORCH_DTYPES=[torch.bfloat16]
register_npu_ci(est_time=500, suite="full-2-npu-a3", nightly=True)

ALL_OTHER_LORA_MODELS = [
    LoRAModelCase(
        # base="meta-llama/Llama-3.1-8B-Instruct",
        # base="/home/weights/Qwen3.5-9B",
        base="/home/weights/Qwen/Qwen3.6-27B-W8A8",
        adaptors=[
            LoRAAdaptor(
                # name="nvidia/llama-3.1-nemoguard-8b-topic-control",
                # name="/home/weights/Qwen3.5-9B-LoRA-new",
                name="/home/weights/qwen3.6-27b-cybersecurity-lora",
                prefill_tolerance=7e-1,
                decode_tolerance=7,
            ),
        ],
        max_loras_per_batch=1,
    ),
    LoRAModelCase(
        # base="meta-llama/Llama-2-7b-hf",
        base="/home/weights/Qwen3.5-4B",
        # adaptors=[LoRAAdaptor(name="winddude/wizardLM-LlaMA-LoRA-7B")],
        adaptors=[
            LoRAAdaptor(
                name="/home/weights/qwen3.5-4b-neo4j-text2cypher-lora",
                prefill_tolerance=3e-1,
                decode_tolerance=3e-1,
            )
        ],
        max_loras_per_batch=2,
    ),
]

CI_LORA_MODELS = [
    LoRAModelCase(
        # base="meta-llama/Llama-3.1-8B-Instruct",
        base="/home/weights/Qwen3.5-4B",
        adaptors=[
            LoRAAdaptor(
                # name="algoprog/fact-generation-llama-3.1-8b-instruct-lora",
                name="/home/weights/qwen3.5-4b-mcat-lora",
                prefill_tolerance=3e-1,
                decode_tolerance=3e-1,
            ),
        ],
        max_loras_per_batch=1,
    ),
]


CI_MULTI_LORA_MODELS = [
    # multi-rank case
    LoRAModelCase(
        base="/home/weights/Qwen3.5-4B",
        adaptors=[
            LoRAAdaptor(
                name="/home/weights/qwen3.5-4b-mcat-lora",
                prefill_tolerance=3e-1,
                decode_tolerance=3e-1,
                rouge_l_tolerance=0.9,
            ),
            LoRAAdaptor(
                name="/home/weights/qwen3.5-4b-neo4j-text2cypher-lora",
                prefill_tolerance=3e-1,
                decode_tolerance=3e-1,
                rouge_l_tolerance=0.9,
            ),
        ],
        max_loras_per_batch=2,
        max_loaded_loras=4,
    ),
]

class TestLoRATP(CustomTestCase):

    def _run_tp_on_model_cases(
        self,
        model_cases: List[LoRAModelCase],
        enable_lora_overlap_loading: Optional[bool] = None,
    ):
        tp_list = [2]  # Define TP sizes to iterate over
        for model_case in model_cases:
            # If skip_long_prompt is True, filter out prompts longer than 1000 characters
            prompts = (
                DEFAULT_PROMPTS
                if not model_case.skip_long_prompt
                else [p for p in DEFAULT_PROMPTS if len(p) < 1000]
            )
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
                    )

    def test_ci_lora_models(self):
        self._run_tp_on_model_cases(CI_LORA_MODELS)

    def test_lora_overlap_loading_ci_lora_models(self):
        self._run_tp_on_model_cases(
            CI_MULTI_LORA_MODELS, enable_lora_overlap_loading=True
        )

    def test_all_lora_models(self):
        # Retain ONLY_RUN check here
        filtered_models = []
        for model_case in ALL_OTHER_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_tp_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
