import multiprocessing as mp
import unittest

import torch

from sglang.test.ascend.lora_utils import (
    # CI_MULTI_LORA_MODELS,
    run_lora_test_one_by_one, LoRAModelCase, LoRAAdaptor,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

PROMPTS = [
    "AI is a field of computer science focused on",
    """
    ### Instruction:
    Tell me about llamas and alpacas
    ### Response:
    Llamas are large, long-necked animals with a woolly coat. They have two toes on each foot instead of three like other camelids.
    ### Question:
    What do you know about llamas?
    ### Answer:
    """,
]
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
                prefill_tolerance=3e-1,
                decode_tolerance=3e-1,
                rouge_l_tolerance=0.9,
            ),
            LoRAAdaptor(
                # name=QWEN3_5_4B_NEO4J_TEXT2CYPHER_LORA_PATH,
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

class TestLoRARadixCache(CustomTestCase):

    def test_lora_radix_cache(self):
        # Here we need a model case with multiple adaptors for testing correctness of radix cache
        model_case = CI_MULTI_LORA_MODELS[0]

        torch_dtype = torch.bfloat16
        max_new_tokens = 32
        batch_prompts = (
            PROMPTS
            if not model_case.skip_long_prompt
            else [p for p in PROMPTS if len(p) < 1000]
        )

        # Test lora with radix cache
        run_lora_test_one_by_one(
            batch_prompts,
            model_case,
            torch_dtype,
            max_new_tokens=max_new_tokens,
            disable_radix_cache=False,
            test_tag="lora-with-radix-cache",
        )

        # Test lora without radix cache
        run_lora_test_one_by_one(
            batch_prompts,
            model_case,
            torch_dtype,
            max_new_tokens=max_new_tokens,
            disable_radix_cache=True,
            test_tag="lora-without-radix-cache",
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
