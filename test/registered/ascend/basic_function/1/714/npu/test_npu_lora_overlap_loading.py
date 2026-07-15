import multiprocessing as mp
import unittest
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.ascend.lora_utils import (
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
                base="/home/weights/Qwen3.5-4BB",
                adaptors=[
                    LoRAAdaptor(
                        name="/home/weights/qwen3.5-4b-mcat-lor",
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
        run_lora_batch_splitting_equivalence_test(
            CI_MULTI_LORA_MODELS,
            enable_lora_overlap_loading=True,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
