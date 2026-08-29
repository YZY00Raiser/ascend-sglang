import os
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_mmlu import TestMMLU
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)

DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH="/home/weights/DeepSeek-V2-Lite-W8A8"
class TestDeepEpDeepseek(GSM8KAscendMixin,TestMMLU,CustomTestCase):
    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    other_args = [
        "--trust-remote-code",
        "--attention-backend",
        "ascend",
        "--tp-size",
        "8",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "auto",
        "--disable-cuda-graph",
        "--dp-size",
        "8",
        "--enable-dp-attention",
        "--chunked-prefill-size",
        "1024",
        "--mem-fraction-static",
        "0.7",
    ]
    env = {
        **os.environ,
        "SGLANG_ENABLE_JIT_DEEPGEMM": "0",
        "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "512",
        "HCCL_BUFFSIZE": "2048",
        "MOE_ENABLE_TOPK_NEG_ONE": "1",
        "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    }

    accuracy = 0.34  # Test GSM8K accuracy ≥0.34
    accuracy_mmlu = 0.38  # Test MMLU accuracy ≥0.38



if __name__ == "__main__":
    unittest.main()
