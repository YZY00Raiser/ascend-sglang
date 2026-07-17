import concurrent.futures
import tempfile
import unittest
from time import sleep

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
#     LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
#     LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
# )
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
BASE = "/home/weights/Qwen3.5-4B"
LORA_A = "/home/weights/Qwen3-4B-lora-v2"
LORA_B = "/home/weights/qwen3.5-4b-neo4j-text2cypher-lora"


class TestLora1(CustomTestCase):
    """Testcase：Verify set the --max-load-rank, --lora-backend parameter, load lora that match the number of ranks,
    inference request successful.

    [Test Category] Parameter
    [Test Target] --max-load-rank, --lora-backend
    """

    lora_a = LORA_A
    lora_b = LORA_B

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-path",
            # f"lora_a={cls.lora_a}",
            f"lora_b={cls.lora_b}",
            "--max-loaded-loras",
            "2",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.3",
            "--log-level",
            "debug",
            "--max-running-requests",
            "2",
            "--max-loras-per-batch",
            "1",
            "--lora-backend",
            "ascend",
        ]
        cls.process = popen_launch_server(
            BASE,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_wait_threshold(self):
        response=requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 128,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)



if __name__ == "__main__":
    unittest.main()
