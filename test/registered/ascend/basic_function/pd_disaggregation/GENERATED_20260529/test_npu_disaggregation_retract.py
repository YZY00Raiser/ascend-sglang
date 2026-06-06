import os
import unittest
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import QWEN3_32B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.few_shot_gsm8k import run_eval
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestNPUDisaggregationSimulatedRetract(PDDisaggregationServerBase):
    """Test retract simulation in disaggregation mode on NPU.

    [Test Category] Functional
    [Test Target] Request retraction and retry mechanism
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        os.environ["SGLANG_TEST_RETRACT"] = "true"
        cls.model = QWEN3_32B_WEIGHTS_PATH
        # Use ascend transfer backend for NPU
        cls.transfer_backend = ["--disaggregation-transfer-backend", "ascend"]
        cls.rdma_devices = []
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_RETRACT")
        super().tearDownClass()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.85",
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            "2",
            "--base-gpu-id",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            "0.85",
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def test_gsm8k(self):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host=f"http://{self.base_host}",
            port=int(self.lb_port),
        )
        metrics = run_eval(args)
        self.assertGreater(metrics["accuracy"], 0.80)


if __name__ == "__main__":
    unittest.main()
