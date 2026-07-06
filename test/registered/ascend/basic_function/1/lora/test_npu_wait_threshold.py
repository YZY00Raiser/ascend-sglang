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
            f"lora_a={cls.lora_a}",
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
            "--lora-drain-wait-threshold",
            "3.0",
            "--base-gpu-id",
            "2",
        ]
        cls.out_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        )
        cls.err_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="err.log"
        )

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )
    def verify_lora(self,message):
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        message = message
        self.assertIn(message, content)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()

    def test_lora_wait_threshold(self):
        def send_request(max_new_tokens, lora_path):
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(send_request, 200, "lora_a")
            future2 = executor.submit(send_request, 300, "lora_b")
            response1 = future1.result()
            response2 = future2.result()
        sleep(3)
        response3 = send_request(32, "lora_a")

        self.assertEqual(response1.status_code, 200)
        self.assertIn("Paris", response1.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        self.assertIn("draining", content)
        self.err_log_file.seek(0)
        self.assertIn("finished draining", content)

'''
class TestLora2(CustomTestCase):
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
            f"lora_a={cls.lora_a}",
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
            "--lora-drain-wait-threshold",
            "3.0",
            "--base-gpu-id",
            "2",
        ]
        cls.out_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        )
        cls.err_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="err.log"
        )

        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )


    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()

    def test_un_lora_wait_threshold(self):
        def send_request(max_new_tokens, lora_path):
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

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(send_request, 200, "lora_a")
            future2 = executor.submit(send_request, 300, "lora_b")
            response1 = future1.result()
            response2 = future2.result()

        sleep(3)
        response3 = send_request(32, "lora_a")

        self.assertEqual(response1.status_code, 200)
        self.assertIn("Paris", response1.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        message = "draining"
        self.assertNotIn(message, content)

'''


if __name__ == "__main__":
    unittest.main()
