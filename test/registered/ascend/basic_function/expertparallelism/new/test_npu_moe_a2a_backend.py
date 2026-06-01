import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH = "/home/weights/DeepSeek-V3.2-W8A8"


class TestMoreRunnerBackendTriton(CustomTestCase):
    """Testcase：Verify set --moe-a2a-backend, the inference request is successfully processed.

    [Test Category] Parameter
    [Test Target]--moe-a2a-backend
    """

    moe_runner_backend = "triton"

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--mem-fraction-static",
                "0.8",
                "--tp-size",
                "16",
                "--expert-parallel-size",
                "16",
                "--enable-eplb",
                "--moe-a2a-backend",
                "ascend_fuseep",  # It is incompatible with eplb
                "--deepep-mode",
                "normal",
            ],
            env={
                "SGLANG_NPUDISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
                "TRANSFORMERS_VERBOSITY": "error",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_moe_runner_backend(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        self.assertEqual(
            response.status_code, 200, "The request status code is not 200."
        )
        self.assertIn(
            "Paris", response.text, "The inference result does not include Paris."
        )


if __name__ == "__main__":
    unittest.main()
