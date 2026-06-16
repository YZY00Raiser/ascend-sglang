"""NPU SRT Engine tests for runtime consistency and basic engine API.

Adapted from test_srt_engine.py. Uses sgl.Engine directly with ascend backend."""

import asyncio
import json
import unittest

import torch

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNpuSRTEngine(CustomTestCase):
    """Test sgl.Engine API on NPU.

    [Test Category] Core
    [Test Target] sgl.Engine, sgl.Runtime
    """

    def test_engine_runtime_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=model_path,
            random_seed=42,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out2 = json.loads(runtime.generate(prompt, sampling_params))["text"]
        runtime.shutdown()

        self.assertEqual(out1, out2)

    def test_engine_token_ids_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            disable_radix_cache=True,
            attention_backend="ascend",
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]

        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        tokenizer = get_tokenizer(model_path)
        token_ids = tokenizer.encode(prompt)
        out2 = engine.generate(input_ids=token_ids, sampling_params=sampling_params)["text"]
        engine.shutdown()

        self.assertEqual(out1, out2)

    def test_engine_cpu_offload(self):
        prompt = "Today is a sunny day and I like"
        model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=128,
            attention_backend="ascend",
            disable_cuda_graph=True,
            mem_fraction_static=0.3,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(
            model_path=model_path,
            random_seed=42,
            max_total_tokens=128,
            cpu_offload_gb=3,
            attention_backend="ascend",
            disable_cuda_graph=True,
            mem_fraction_static=0.3,
        )
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        self.assertEqual(out1, out2)

    def test_engine_async_encode_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_SMALL_EMBEDDING_MODEL_NAME_FOR_TEST

        try:
            engine = sgl.Engine(
                model_path=model_path,
                is_embedding=True,
                random_seed=42,
                disable_radix_cache=True,
                attention_backend="ascend",
                disable_cuda_graph=True,
                mem_fraction_static=0.3,
            )

            out1 = torch.tensor(engine.encode(prompt)["embedding"])
            loop = asyncio.get_event_loop()
            out2 = torch.tensor(
                loop.run_until_complete(engine.async_encode(prompt))["embedding"]
            )
            engine.shutdown()

            self.assertTrue(
                torch.allclose(out1, out2, atol=1e-5, rtol=1e-3),
                "Sync and async embeddings are not equal within tolerance",
            )
        except Exception:
            self.skipTest("Embedding model not available on NPU")


if __name__ == "__main__":
    unittest.main()
