import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,

)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
git clone https://huggingface.co/kingabzpro/Llama-4-Scout-17B-16E-Instruct-Medical-ChatBot
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestLoraBasicFunction(CustomTestCase):
    """Testcase：Verify the use llama4 model with lora, inference request succeeded.

    [Test Category] Parameter
    [Test Target] --enable-lora, --lora-path,
    """

    lora_a = ""

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            "--lora-target-modules",
            "all",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.9",
            "--disable-radix-cache",
        ]
        cls.process = popen_launch_server(
            LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lama4_lora(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)


if __name__ == "__main__":
    unittest.main()
