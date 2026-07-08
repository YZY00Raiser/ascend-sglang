import unittest

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_npu_ci(est_time=600, suite="full-4-npu-a3", nightly=True)

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH,
)


class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):
    model =  QWEN3_NEXT_80B_A3B_INSTRUCT_WEIGHTS_PATH
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


if __name__ == "__main__":
    unittest.main()
