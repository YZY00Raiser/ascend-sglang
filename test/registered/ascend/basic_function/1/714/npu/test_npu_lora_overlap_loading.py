import multiprocessing as mp
import unittest
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.lora_utils import (
    # CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test, LoRAModelCase, LoRAAdaptor,
)
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestLoRAOverlapLoading(CustomTestCase):

    def test_ci_lora_models_batch_splitting(self):
        CI_MULTI_LORA_MODELS = [
            # multi-rank case
            LoRAModelCase(
                base="/home/weights/Llama-3.2-1B-Instruct",
                adaptors=[
                    LoRAAdaptor(
                        name="/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora",
                        prefill_tolerance=1e-1,
                        rouge_l_tolerance=0.9,
                    ),
                    LoRAAdaptor(
                        name="/home/weights/codelion/FastLlama-3.2-LoRA",
                        prefill_tolerance=3e-1,
                        rouge_l_tolerance=0.9,
                    ),
                ],
                max_loras_per_batch=2,
                max_loaded_loras=4,
            ),
        ]
        run_lora_batch_splitting_equivalence_test(
            CI_MULTI_LORA_MODELS,
            enable_lora_overlap_loading=True,
            attention_backend="ascend",
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
