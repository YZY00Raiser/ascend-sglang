import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import sglang as sgl
from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)

DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH="/home/weights/DeepSeek-V3.2-W8A8"
class _BaseTestDynamicEPLB(CustomTestCase):
    """Test dynamic EPLB functionality on NPU.

    [Test Category] Parameter
    [Test Target] --enable-eplb, --ep-num-redundant-experts, dynamic rebalancing
    """
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False),
            envs.SGLANG_EXPERT_LOCATION_UPDATER_CANARY.override(True),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp-size",
                    "16",
                    "--enable-dp-attention",
                    "--dp-size",
                    "2",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--disable-cuda-graph",
                    "--enable-eplb",
                    "--ep-num-redundant-experts",
                    16,
                    "--eplb-rebalance-num-iterations",
                    50,
                    "--expert-distribution-recorder-buffer-size",
                    50,
                    "--quantization",
                    "modelslim",
                    "--mem-fraction-static",
                    0.82,
                    "--context-length",
                    40960,
                    "--max-prefill-tokens",
                    40960,
                    "--max-total-tokens",
                    40960,
                    "--expert-distribution-recorder-mode",
                    "stat",
                    "--ep-dispatch-algorithm",
                    "static",
                    "--disable-radix-cache",
                    *cls.extra_args,
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
                    "SGLANG_DEEPEP_BF16_DISPATCH": "1",
                    "TRANSFORMERS_VERBOSITY": "error",
                    **os.environ,
                },
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=128,
            num_threads=64,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.85)


class TestDynamicEPLBSimple(_BaseTestDynamicEPLB):
    """Test dynamic EPLB basic functionality on NPU.

    [Test Category] EPLB
    [Test Target] --enable-eplb, dynamic rebalancing
    """
    pass


class TestDynamicEPLBMultiChunk(_BaseTestDynamicEPLB):
    """Test dynamic EPLB with multi-chunk rebalancing on NPU.

    [Test Category] EPLB
    [Test Target] --enable-eplb, --eplb-rebalance-layers-per-chunk
    """
    extra_args = ["--eplb-rebalance-layers-per-chunk", "1"]


class TestStaticEPLB(CustomTestCase):
    """Test static EPLB with expert distribution recording and initialization on NPU.

    [Test Category] Parameter
    [Test Target] --enable-eplb, expert distribution recorder, init_expert_location
    """
    def test_save_expert_distribution_and_init_expert_location(self):
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)

        with tempfile.TemporaryDirectory() as tmp_dir:
            engine_kwargs = dict(
                model_path=DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
                trust_remote_code=True,
                ep_num_redundant_experts=16,
                enable_dp_attention=True,
                dp_size=2,
                moe_a2a_backend="deepep",
                disable_cuda_graph=True,
                expert_distribution_recorder_mode="stat",
                tp_size=16,
                log_level="info",
                quantization="modelslim",
                mem_fraction_static=0.82,
                context_length=40960,
                max_prefill_tokens=40960,
                max_total_tokens=40960,
                disable_radix_cache=True,
            )

            envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.set(tmp_dir)
            engine = sgl.Engine(
                **engine_kwargs,
                disable_overlap_schedule=True,
            )
            engine.start_expert_distribution_record()
            self._assert_engine_generate_correct(engine)

            engine.dump_expert_distribution_record()
            snapshot_path = list(Path(tmp_dir).glob("*.pt"))[0]
            assert snapshot_path is not None

            engine.shutdown()
            del engine

            engine = sgl.Engine(
                **engine_kwargs,
                init_expert_location=str(snapshot_path),
                port=21000,
                ep_dispatch_algorithm="static",
            )
            self._assert_engine_generate_correct(engine)
            engine.shutdown()
            del engine

    def _assert_engine_generate_correct(self, engine: sgl.Engine):
        output = engine.generate(
            prompt=["1+1=2, 2+2=4", "One plus one is two, two plus two is four"],
            sampling_params=dict(max_new_tokens=8, temperature=0.0),
        )
        self.assertEqual(
            [x["text"] for x in output],
            [", 4+4=8,", ", four plus four is eight, eight"],
        )


if __name__ == "__main__":
    unittest.main()
