import os
import unittest
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.kl_divergence_kit import KLDivergenceMixin
from sglang.test.kits.prefix_cache_branching_kit import PrefixCacheBranchingMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.server_fixtures.default_fixture import (
    DefaultServerBase,
    openai_api_env,
)
from sglang.test.test_utils import (
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=1200, suite="full-4-npu-a3", nightly=True)

_NPU_ENV = {
    **os.environ,
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "ASCEND_MF_STORE_URL": "tcp://127.0.0.1:24666",
    "HCCL_BUFFSIZE": "200",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "24",
    "USE_VLLM_CUSTOM_ALLREDUCE": "1",
    "HCCL_EXEC_TIMEOUT": "200",
    "STREAMS_PER_DEVICE": "32",
    "AUTO_USE_UC_MEMORY": "0",
    "P2P_HCCL_BUFFSIZE": "20",
    # NPU spec v2 path required for NEXTN MTP (see Ascend reference:
    # test_npu_qwen3_next_80b_w8a8_2p_in3k5_out1k5_50ms.py).
    "SGLANG_ENABLE_SPEC_V2": "1",
}

QWEN3_NEXT_MODEL="/home/weights/Qwen3-Next-80B-A3B-Instruct"
class TestQwen3NextMTPV2(GSM8KMixin, KLDivergenceMixin, DefaultServerBase):


    @classmethod
    def setUpClass(cls):
        assert cls.model is not None, "Please set cls.model in subclass"
        with openai_api_env(cls.api_key):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=cls.timeout,
                other_args=cls.other_args,
                env=_NPU_ENV,
            )
    model = QWEN3_NEXT_MODEL
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

if __name__ == "__main__":
    unittest.main()
