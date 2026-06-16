"""NPU Engine child PID test.

Verifies that launching an Engine on NPU exposes the PIDs of all child processes
and that those PIDs correspond to live processes."""

import os
import unittest

import psutil

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


class TestNpuEngineChildPids(CustomTestCase):
    """Test Engine child PID management on NPU.

    [Test Category] Core
    [Test Target] Engine.get_all_child_pids()
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            random_seed=42,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_get_all_child_pids_returns_live_pids(self):
        pids = self.engine.get_all_child_pids()
        self.assertIsInstance(pids, list)
        self.assertGreater(len(pids), 0, "Expected at least one child PID")
        for pid in pids:
            self.assertIsInstance(pid, int)
            self.assertTrue(
                psutil.pid_exists(pid),
                f"PID {pid} does not correspond to a running process",
            )

    def test_child_pids_include_scheduler_and_detokenizer(self):
        pids = self.engine.get_all_child_pids()
        self.assertGreaterEqual(
            len(pids),
            2,
            "Expected at least 2 child PIDs (scheduler + detokenizer)",
        )

    def test_child_pids_no_duplicates(self):
        pids = self.engine.get_all_child_pids()
        self.assertEqual(
            len(pids),
            len(set(pids)),
            f"Duplicate PIDs found: {pids}",
        )


if __name__ == "__main__":
    unittest.main()
