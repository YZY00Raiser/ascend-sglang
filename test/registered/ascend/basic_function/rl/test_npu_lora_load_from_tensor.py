import json
import os
import unittest

from safetensors.torch import load_file

import sglang as sgl
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=150, suite="nightly-2-npu-a3", nightly=True)

MODEL_PATH = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
LORA_PATH = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
TEST_PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 16


class TestNPULoRALoadFromTensor(CustomTestCase):
    """Test LoRA load from tensor on NPU.

    [Test Category] RL LoRA
    [Test Target] Engine.load_lora_adapter_from_tensors
    """

    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=MODEL_PATH,
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            mem_fraction_static=0.6,
            log_level="error",
        )

        # Load LoRA from local path
        cls.lora_tensors = load_file(
            os.path.join(LORA_PATH, "adapter_model.safetensors")
        )
        with open(os.path.join(LORA_PATH, "adapter_config.json"), "r") as f:
            cls.lora_config_dict = json.load(f)

    def test_lora_lru_eviction(self):
        # Test LRU eviction using the class-level engine
        # Load multiple LoRA adapters to trigger LRU eviction
        TEST_LORA_COUNT = 10
        for i in range(TEST_LORA_COUNT):
            result = self.engine.load_lora_adapter_from_tensors(
                lora_name=f"tool_calling_lora_evict_{i}",
                tensors=self.lora_tensors,
                config_dict=self.lora_config_dict,
            )
            self.assertTrue(
                result.success,
                f"Failed to load LoRA adapter {i}: {result.error_message}",
            )

        # Verify that the last 8 adapters are loaded (LRU eviction)
        EXPECTED_LORA_ADAPTERS = [
            "tool_calling_lora_evict_2",
            "tool_calling_lora_evict_3",
            "tool_calling_lora_evict_4",
            "tool_calling_lora_evict_5",
            "tool_calling_lora_evict_6",
            "tool_calling_lora_evict_7",
            "tool_calling_lora_evict_8",
            "tool_calling_lora_evict_9",
        ]
        EXPECTED_LORA_COUNT = 8
        self.assertEqual(
            len(result.loaded_adapters),
            EXPECTED_LORA_COUNT,
        )
        self.assertEqual(
            list(result.loaded_adapters.keys()),
            EXPECTED_LORA_ADAPTERS,
        )

    def test_lora_e2e_load_from_tensor_params(self):
        result = self.engine.load_lora_adapter_from_tensors(
            lora_name="tool_calling_lora",
            tensors=self.lora_tensors,
            config_dict=self.lora_config_dict,
        )
        self.assertTrue(
            result.success,
            f"Failed to load LoRA from tensors: {result.error_message}",
        )

        output_without_lora = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
        )

        output_lora = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
            lora_path=["tool_calling_lora"],
        )

        # Verify LoRA produces different output than base model
        self.assertNotEqual(
            output_without_lora[0]["text"],
            output_lora[0]["text"],
            "LoRA should produce different output than base model",
        )

    def test_lora_load_unload_load_from_tensor_params(self):
        result = self.engine.load_lora_adapter_from_tensors(
            lora_name="tool_calling_lora_multiple",
            tensors=self.lora_tensors,
            config_dict=self.lora_config_dict,
        )
        self.assertTrue(
            result.success,
            f"Failed to load LoRA from tensors: {result.error_message}",
        )

        result = self.engine.unload_lora_adapter("tool_calling_lora_multiple")
        self.assertTrue(
            result.success, f"Failed to unload LoRA: {result.error_message}"
        )
        with self.assertRaises(ValueError):
            self.engine.generate(
                prompt=[TEST_PROMPT],
                sampling_params={
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "temperature": 0.0,
                },
                lora_path=["tool_calling_lora_multiple"],
            )

        result_again = self.engine.load_lora_adapter_from_tensors(
            lora_name="tool_calling_lora_multiple",
            tensors=self.lora_tensors,
            config_dict=self.lora_config_dict,
        )
        self.assertTrue(
            result_again.success,
        )
        output_lora_loaded_again = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
            lora_path=["tool_calling_lora_multiple"],
        )

        # Verify output is generated successfully after reload
        self.assertIsNotNone(output_lora_loaded_again[0]["text"])

    def test_lora_e2e_load_from_flattened_bucket(self):
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

        named_tensors = list(self.lora_tensors.items())
        bucket = FlattenedTensorBucket(named_tensors=[(n, t) for n, t in named_tensors])
        bucket_dict = {
            "flattened_tensor": bucket.get_flattened_tensor(),
            "metadata": bucket.get_metadata(),
        }
        serialized = MultiprocessingSerializer.serialize(bucket_dict, output_str=True)

        result = self.engine.load_lora_adapter_from_tensors(
            lora_name="tool_calling_lora_flattened",
            tensors=serialized,
            config_dict=self.lora_config_dict,
            load_format="flattened_bucket",
        )
        self.assertTrue(result.success, f"Failed: {result.error_message}")

        output = self.engine.generate(
            prompt=[TEST_PROMPT],
            sampling_params={"max_new_tokens": MAX_NEW_TOKENS, "temperature": 0.0},
            lora_path=["tool_calling_lora_flattened"],
        )
        # Verify output is generated successfully
        self.assertIsNotNone(output[0]["text"])

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()


if __name__ == "__main__":
    unittest.main()
