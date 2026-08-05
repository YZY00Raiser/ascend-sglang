import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval as run_gsm8k
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH="/home/weights/DeepSeek-V3.2-W8A8"
register_npu_ci(est_time=400, suite="full-16-npu-a3", nightly=True)


class TestDeepEpDeepseekV32(CustomTestCase):
    """Testcase: Verify that for the DeepSeek V3.2 model in the single-machine colocation scenario,
    its inference accuracy on the MMLU and GSM8K dataset meets the preset standard when the parameter --deepep-mode auto is configured.

    [Test Category] Expert Parallelism
    [Test Target] --moe-a2a-backend deepep;--deepep-mode
    """

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=6000,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "16",
                "--quantization",
                "modelslim",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-mode",
                "auto",
                "--mem-fraction-static",
                0.86,
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--context-length",
                40960,
                "--max-prefill-tokens",
                40960,
                "--max-total-tokens",
                40960,
                "--moe-dense-tp-size",
                "1",
                "--enable-dp-lm-head",
                "--ep-num-redundant-experts",
                "32",
                "--ep-dispatch-algorithm",
                "dynamic",
                "--eplb-algorithm",
                "deepseek",
                "--cuda-graph-bs",
                "64",
                "--max-running-requests",
                "512",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--model-loader-extra-config",
                '{"enable_multithread_load": true,"num_threads": 64}'
            ],
            env={
                "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
                "STREAMS_PER_DEVICE": "32",
                "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "16",
                "HCCL_BUFFSIZE": "1600",
                "HCCL_OP_EXPANSION_MODE": "AIV",
                "SGLANG_NPU_USE_MLAPO": "0",
                "SGLANG_NPU_USE_MULTI_STREAM": "1",
                "TASK_QUEUE_ENABLE": "0",
                "TRANSFORMERS_VERBOSITY": "error",
                "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
                **os.environ,
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        expect_score = 0.85
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=32,
        )
        print("Starting mmlu test...")
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], expect_score)

    def test_gsm8k(self):
        expect_accuracy = 0.95
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            timeout=60000,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        print("Starting gsm8k test...")
        metrics = run_gsm8k(args)
        self.assertGreaterEqual(
            metrics["accuracy"],
            expect_accuracy,
            f'Accuracy of {self.model} is {str(metrics["accuracy"])}, is lower than {expect_accuracy}',
        )
        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k:\n"
            f"accuracy={metrics['score']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 1.85)


if __name__ == "__main__":
    unittest.main()
