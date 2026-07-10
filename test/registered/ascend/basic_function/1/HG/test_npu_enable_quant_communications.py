import tempfile
import unittest
from types import SimpleNamespace

import requests

# from sglang.test.ascend.test_ascend_utils import
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
# model="/root/.cache/modelscope/hub/models/Eco-Tech/Qwen3.5-35B-A3B-w8a8-mtp"
model = "/home/weights/Qwen3.5-35B-A3B-w8a8-mtp"
register_npu_ci(est_time=200, suite="full-4-npu-a3", nightly=True)


class TestDtypeAuto(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.out_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="out.log"
        )
        cls.err_log_file = tempfile.NamedTemporaryFile(
            mode="w+", delete=True, suffix="err.log"
        )
        cls.model = model
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--mem-fraction-static",
                "0.8",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-quant-communications",
                "--log-level",
                "info",
            ],
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")
        self.assertGreater(metrics["score"], 0.74)
        self.err_log_file.seek(0)
        content = self.err_log_file.read()
        error_message = "enable_quant_communications=True"
        self.assertIn(error_message, content)

if __name__ == "__main__":
    unittest.main()
