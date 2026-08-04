import multiprocessing as mp
import unittest

import torch

from sglang.test.ascend.lora_utils import (
    # CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test, LoRAModelCase, LoRAAdaptor,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=300, suite="full-1-npu-a3", nightly=True)
CI_MULTI_LORA_MODELS = [
    # multi-rank case
    LoRAModelCase(
        # base=QWEN3_5_4B_WEIGHTS_PATH,
        base="/home/weights/Qwen3.5-4B",
        adaptors=[
            LoRAAdaptor(
                # name=QWEN3_5_4B_MCAT_LORA_PATH,
                name="/home/weights/qwen3.5-4b-mcat-lora",
                prefill_tolerance=1e-1,
                rouge_l_tolerance=0.9,
            ),
            LoRAAdaptor(
                # name=QWEN3_5_4B_NEO4J_TEXT2CYPHER_LORA_PATH,
                name="/home/weights/qwen3.5-4b-neo4j-text2cypher-lora",
                prefill_tolerance=3e-1,
                rouge_l_tolerance=0.9,
            ),
        ],
        max_loras_per_batch=2,
        max_loaded_loras=4,
    ),
]

class TestLoRAOverlapLoading(CustomTestCase):

    def test_ci_lora_models_batch_splitting(self):
        run_lora_batch_splitting_equivalence_test(
            CI_MULTI_LORA_MODELS,
            enable_lora_overlap_loading=True,
            torch_dtype=torch.bfloat16,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
