import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)

QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH="/home/weights/Qwen/Qwen3-30B-A3B-Instruct-2507"
class TestEPLBDispatchAlgorithmStatic(CustomTestCase):
    """Testcase: Verify that the model accuracy remains uncompromised when the parameter --ep-dispatch-algorithm is configured.

    [Test Category] Parameter
    [Test Target] --ep-dispatch-algorithm
    """
    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    ep_dispatch_algorithm = "static"

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            cls.model,
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
                "2",
                "--expert-parallel-size",
                "2",
                "--enable-eplb",
                "--ep-dispatch-algorithm",
                cls.ep_dispatch_algorithm,
            ],
            env={
                "SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT": "1",
                "HCCL_BUFFSIZE": "1024",
                "TRANSFORMERS_VERBOSITY": "error",
            },
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            max_new_tokens=512,
            base_url=DEFAULT_URL_FOR_TEST,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            num_examples=200,
            num_threads=128,
            num_shots=5,
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.90)


class TestEPLBDispatchAlgorithmDynamic(TestEPLBDispatchAlgorithmStatic):
    ep_dispatch_algorithm = "dynamic"


class TestEPLBDispatchAlgorithmFake(TestEPLBDispatchAlgorithmStatic):
    ep_dispatch_algorithm = "fake"


if __name__ == "__main__":
    unittest.main()
