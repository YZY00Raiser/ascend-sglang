import os
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_mmlu import TestMMLU
from sglang.test.ascend.run_eval import run_eval as run_ascend_eval, run_eval
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)


class TestDeepEpDeepseek(GSM8KAscendMixin, TestMMLU, CustomTestCase):
    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    # accuracy = 0.34
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

    accuracy_mmlu = 0.85  # Test MMLU accuracy ≥0.38

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=1200,
            num_threads=1200,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")

        self.assertGreater(metrics["score"], 0.34)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k:\n"
            f"accuracy={metrics['score']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 1.85)


if __name__ == "__main__":
    unittest.main()
