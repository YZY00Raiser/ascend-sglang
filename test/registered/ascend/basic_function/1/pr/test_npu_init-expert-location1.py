import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
)
register_npu_ci(est_time=200, suite="full-2-npu-a3", nightly=True)
# Read the content of the expert distribution file
with open("qwen3_30b_a3b_random_no_redundant.json", "r") as f:
    init_expert_location = f.read()


class TestInitExpertLocationString(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the /home/weights/Qwen3-30B-A3B model on the GSM8K dataset is no less than 0.90.

    [Test Category] Parameters
    [Test Target] --init-expert-location
    """

    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.7,
        "--expert-distribution-recorder-mode",
        "stat",
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
        init_expert_location  # 直接使用文件内容，而非文件路径
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
        "SGLANG_LOG_EXPERT_LOCATION_METADATA": "1"  # 可以从打屏看到更多信息，用不到的话可以删掉
    }


class TestInitExpertLocationJson(GSM8KAscendMixin, CustomTestCase):
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
        # "expert_distribution_recorder_1783072007.920626.pt",
        "expert_distribution_recorder.json"
        # "expert_distribution_recorder.pt",
        # "qwen3_30b_a3b_random_no_redundant.json",
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
    }


class TestInitExpertLocationPt(GSM8KAscendMixin, CustomTestCase):
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
        "expert_distribution_recorder.pt",
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
    }


class TestInitExpertLocationTrivial(GSM8KAscendMixin, CustomTestCase):
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
        "trivial",
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
    }


if __name__ == "__main__":
    unittest.main()
