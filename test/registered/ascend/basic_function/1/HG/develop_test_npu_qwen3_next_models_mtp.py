import unittest

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase, openai_api_env
from sglang.test.test_utils import popen_launch_server

register_npu_ci(est_time=600, suite="full-8-npu-a3", nightly=True)

# from sglang.test.ascend.test_ascend_utils import (
#     QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
# )

QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH="/home/weights/Qwen3-Next-80B-A3B-Instruct"
class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    model =  QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    gsm8k_accuracy_thres = 0.93
    # kl_div_thres = 0.0035
    kl_div_thres = 0.02
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--mamba-ssm-dtype",
        "bfloat16",
        "--mem-fraction-static",
        "0.75",
        "--tp-size",
        "8",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
    ]

    @classmethod
    def setUpClass(cls):
        assert cls.model is not None, "Please set cls.model in subclass"

        # Set OpenAI API key and base URL environment variables.
        # Needed for lmm-evals to work.
        with openai_api_env(cls.api_key):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=cls.other_args,
                env={
                    "SGLANG_ENABLE_SPEC_V2": "1",
                }
            )




if __name__ == "__main__":
    unittest.main()
