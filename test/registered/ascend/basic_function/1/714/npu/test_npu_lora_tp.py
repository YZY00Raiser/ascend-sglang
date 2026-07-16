import multiprocessing as mp
import os
import unittest
from typing import List, Optional

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.lora_utils import (
    ALL_OTHER_LORA_MODELS,
    CI_LORA_MODELS,
    CI_MULTI_LORA_MODELS,
    DEFAULT_PROMPTS,
    TORCH_DTYPES,
    LoRAModelCase,
    run_lora_test_one_by_one,
)
from sglang.test.test_utils import CustomTestCase, is_in_ci

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


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
                        attention_backend="fa3",
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
