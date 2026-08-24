import os
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_mmlu import TestMMLU
from sglang.test.ascend.run_eval import run_eval
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="full-8-npu-a3", nightly=True)

DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH="/home/weights/DeepSeek-V2-Lite-W8A8"
class TestDeepEpDeepseek(GSM8KAscendMixin, TestMMLU, CustomTestCase):
    model = DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
    other_args = [
        "--disable-overlap-schedule",
        "--trust-remote-code",
        "--tp",
        "2",
        "--enable-dp-attention",
        "--dp",
        "2",
        "--moe-dense-tp-size",
        "1",
        "--enable-dp-lm-head",
        "--moe-a2a-backend",
        "deepep",
        "--ep-num-redundant-experts",
        "32",
        "--ep-dispatch-algorithm",
        "dynamic",
        "--eplb-algorithm",
        "deepseek",
        "--cuda-graph-bs",
        "64",
        "--max-running-requests",
        "512",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "1",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "2",
        "--disable-radix-cache",
        "--model-loader-extra-config",
        '{"enable_multithread_load": true,"num_threads": 64}',
        "--mem-fraction-static",
        "0.55",
        "--attention-backend",
        "ascend",
        "--quantization",
        "modelslim",
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
