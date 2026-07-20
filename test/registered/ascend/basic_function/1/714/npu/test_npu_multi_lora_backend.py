import multiprocessing as mp
import os
import unittest

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.lora_utils import (
    ALL_OTHER_MULTI_LORA_MODELS,
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
    run_lora_multiple_batch_on_model_cases,
)
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


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
