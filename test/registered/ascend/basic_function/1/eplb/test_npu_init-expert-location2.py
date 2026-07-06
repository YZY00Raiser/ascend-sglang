import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH="/home/weights/Qwen/Qwen3-30B-A3B-Instruct-2507"


class TestQwen330B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the /home/weights/Qwen3-30B-A3B model on the GSM8K dataset is no less than 0.90.

    [Test Category] Model
    [Test Target] /home/weights/Qwen3-30B-A3B
    """

    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.7,
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
        "--ep-dispatch-algorithm",
        "static",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--init-expert-location",
        #"/tmp/pt/expert_distribution_recorder_1783072007.920626.pt",
        #"/home/d30060301/pt/expert_distribution_recorder.pt"
        "expert_distribution_recorder.json"
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
    }


if __name__ == "__main__":
    unittest.main()
