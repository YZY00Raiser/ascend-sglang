import unittest

from sglang.jit_kernel.benchmark.marker import skip
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase, openai_api_env
from sglang.test.test_utils import popen_launch_server

# from sglang.test.ascend.test_ascend_utils import (
#     QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
# )
register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)

QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH="/home/weights/Qwen3-Next-80B-A3B-Instruct"

@unittest.skip("")
class TestQwen3NextMTPTopk(
    GSM8KMixin, KLDivergenceMixin, PrefixCacheBranchingMixin, DefaultServerBase
):
    # topk > 1 (tree) MTP on a hybrid-GDN model, on spec v2: the tree-aware mamba
    # state update lives in the spec v2 verify path, so mamba + topk > 1 no longer
    # falls back to spec v1.
    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    cache_chunk_size = 64
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.008
    other_args = [
        "--trust-remote-code",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "4",
        "--speculative-num-draft-tokens",
        "8",
        "--mem-fraction-static",
        "0.75",
        "--tp",
        "8",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--attention-backend",
        "ascend",
    ]


class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    model = QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
    gsm8k_accuracy_thres = 0.93
    kl_div_thres = 0.0035
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
        "--mem-fraction-static",
        "0.8",
        "--tp",
        "4",
        "--chunked-prefill-size",
        "2048",
        "--mamba-scheduler-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "128",
        "--attention-backend",
        "ascend",
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
