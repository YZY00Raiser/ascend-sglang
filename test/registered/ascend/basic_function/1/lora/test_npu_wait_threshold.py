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
LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = "/home/weights/Llama-3.2-1B-Instruct"
LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH = "/home/weights/codelion/FastLlama-3.2-LoRA"


class TestLora1(CustomTestCase):
    """Testcase：Verify set the --max-load-rank, --lora-backend parameter, load lora that match the number of ranks,
    inference request successful.

    [Test Category] Parameter
    [Test Target] --max-load-rank, --lora-backend
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH
    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_b}",
            f"lora_3={cls.lora_a}",
            "--lora-backend",
            "ascend",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--max-loras-per-batch",
            "2",
            "--max-running-requests",
            "2",
            "--lora-drain-wait-threshold",
            "3.0",
            "--base-gpu-id",
            "2",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(lora_path, max_new_tokens):
        return requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "lora_path": lora_path,
            },
        )

    def test_lora_max_lora_rank(self):


        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(self._generate, "lora_1", 1000)
            future2 = executor.submit(self._generate, "lora_2", 1500)
            response1 = future1.result()
            response2 = future2.result()
        sleep(3)
        response3 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_3",
            },
        )

        self.assertEqual(response1.status_code, 200)
        self.assertIn("Paris", response1.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)



'''
class TestLora2(CustomTestCase):
    """Testcase：Verify set the --max-load-rank, --lora-backend parameter, load lora that match the number of ranks,
    inference request successful.

    [Test Category] Parameter
    [Test Target] --max-load-rank, --lora-backend
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-path",
            f"lora_1={cls.lora_a}",
            f"lora_2={cls.lora_a}",
            f"lora_3={cls.lora_a}",
            "--lora-backend",
            "ascend",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            # "--experts-shared-outer-loras"
            "--lora-drain-wait-threshold",
            "0"
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_max_lora_rank(self):
        response1 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_1",
            },
        )

        response2 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_2",
            },
        )
        sleep(3)
        response3 = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_3",
            },
        )

        self.assertEqual(response1.status_code, 200)
        self.assertIn("Paris", response1.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
'''

if __name__ == "__main__":
    unittest.main()
