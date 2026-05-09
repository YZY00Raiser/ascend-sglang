# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
NPU LoRA Dynamic Update Test

This test verifies dynamic LoRA adapter loading/unloading via HTTP API on NPU.
Test cases are ported from GPU test_lora_update.py for NPU compatibility.

[Test Category] Integration
[Test Target] /v1/load_lora_adapter, /v1/unload_lora_adapter APIs on NPU
"""

import json
import multiprocessing as mp
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci

# Model paths
LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = "/home/weights/LLM-Research/Llama-3.2-1B-Instruct"
LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH = "/home/weights/codelion/FastLlama-3.2-LoRA"
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    est_time=400,
    suite="nightly-2-npu-a3",
    nightly=True,
)

PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
]

MEM_FRACTION_STATIC = 0.6


class OperationType(Enum):
    LOAD = "load"
    UNLOAD = "unload"
    FORWARD = "forward"


@dataclass
class Operation:
    """Operation definition for LoRA dynamic update test."""

    type: OperationType
    data: Optional[Any]
    expected_error: Optional[str] = None


@dataclass
class TestCase:
    """Test case definition for LoRA dynamic update."""

    description: str
    base: str
    max_loras_per_batch: int
    all_adapters: List[str]
    op_sequence: List[Operation]
    initial_adapters: Optional[List[str]] = None
    enable_lora: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    lora_target_modules: Optional[List] = None
    max_new_tokens: int = 32
    max_loaded_loras: Optional[int] = None


def create_batch_data(adapters):
    """Create batch data for inference."""
    if not isinstance(adapters, list):
        adapters = [adapters]
    return [(prompt, adapter) for prompt in PROMPTS for adapter in adapters]


class TestNpuLoRAUpdate(CustomTestCase):
    """Test dynamic LoRA adapter loading/unloading on NPU."""

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
    lora_b = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    def _run_test_case(self, test_case: TestCase):
        """Execute a single test case."""
        print(f"\n{'='*80}")
        print(f"Running test case: {test_case.description}")
        print(f"{'='*80}")

        # Prepare server arguments
        other_args = [
            "--max-loras-per-batch",
            str(test_case.max_loras_per_batch),
            "--max-model-len",
            "4096",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--mem-fraction-static",
            str(MEM_FRACTION_STATIC),
        ]

        if test_case.enable_lora:
            other_args.append("--enable-lora")

        if test_case.max_lora_rank:
            other_args.extend(["--max-lora-rank", str(test_case.max_lora_rank)])

        if test_case.lora_target_modules:
            other_args.extend(
                ["--lora-target-modules", ",".join(test_case.lora_target_modules)]
            )

        if test_case.max_loaded_loras:
            other_args.extend(
                ["--max-loaded-loras", str(test_case.max_loaded_loras)]
            )

        if test_case.initial_adapters:
            for adapter in test_case.initial_adapters:
                other_args.extend(["--lora-path", adapter])

        # Start server
        process = popen_launch_server(
            test_case.base,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

        try:
            self._execute_operations(test_case)
        finally:
            kill_process_tree(process.pid)

    def _execute_operations(self, test_case: TestCase):
        """Execute operation sequence."""
        base_url = DEFAULT_URL_FOR_TEST

        for op in test_case.op_sequence:
            if op.type == OperationType.LOAD:
                self._load_lora(base_url, op.data, op.expected_error)
            elif op.type == OperationType.UNLOAD:
                self._unload_lora(base_url, op.data, op.expected_error)
            elif op.type == OperationType.FORWARD:
                self._run_inference(base_url, op.data, test_case.max_new_tokens)

    def _load_lora(self, base_url: str, lora_path: str, expected_error: Optional[str]):
        """Load a LoRA adapter via API."""
        print(f"\n[LOAD] Loading LoRA: {lora_path}")

        response = requests.post(
            f"{base_url}/v1/load_lora_adapter",
            json={"lora_name": lora_path, "lora_path": lora_path},
        )

        if expected_error:
            self.assertNotEqual(
                response.status_code,
                200,
                f"Expected error containing '{expected_error}' but got success",
            )
            self.assertIn(
                expected_error.lower(),
                response.text.lower(),
                f"Expected error message containing '{expected_error}' but got: {response.text}",
            )
            print(f"  Expected error occurred: {expected_error}")
        else:
            self.assertEqual(
                response.status_code,
                200,
                f"Failed to load LoRA: {response.text}",
            )
            print(f"  Successfully loaded")

    def _unload_lora(self, base_url: str, lora_path: str, expected_error: Optional[str]):
        """Unload a LoRA adapter via API."""
        print(f"\n[UNLOAD] Unloading LoRA: {lora_path}")

        response = requests.post(
            f"{base_url}/v1/unload_lora_adapter",
            json={"lora_name": lora_path},
        )

        if expected_error:
            self.assertNotEqual(
                response.status_code,
                200,
                f"Expected error containing '{expected_error}' but got success",
            )
            self.assertIn(
                expected_error.lower(),
                response.text.lower(),
                f"Expected error message containing '{expected_error}' but got: {response.text}",
            )
            print(f"  Expected error occurred: {expected_error}")
        else:
            self.assertEqual(
                response.status_code,
                200,
                f"Failed to unload LoRA: {response.text}",
            )
            print(f"  Successfully unloaded")

    def _run_inference(
        self, base_url: str, batch_data: List[tuple], max_new_tokens: int
    ):
        """Run inference with specified LoRA adapters."""
        print(f"\n[FORWARD] Running inference with {len(batch_data)} prompts")

        for prompt, lora_path in batch_data:
            request_data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            }
            if lora_path:
                request_data["lora_path"] = lora_path

            response = requests.post(f"{base_url}/generate", json=request_data)
            self.assertEqual(response.status_code, 200, f"Inference failed: {response.text}")

        print(f"  Inference completed for all prompts")

    def test_npu_lora_dynamic_load_unload(self):
        """Test dynamic LoRA loading and unloading with initial adapters."""
        test_case = TestCase(
            description="Dynamic LoRA update with initial lora_paths on NPU",
            base=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            max_loras_per_batch=2,
            all_adapters=[self.lora_a, self.lora_b],
            initial_adapters=[
                f"lora_a={self.lora_a}",
                f"lora_b={self.lora_b}",
            ],
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            op_sequence=[
                Operation(
                    type=OperationType.LOAD,
                    data="lora_a",
                    expected_error="already loaded",
                ),
                Operation(
                    type=OperationType.UNLOAD,
                    data="lora_a",
                ),
                Operation(
                    type=OperationType.LOAD,
                    data="lora_a",
                ),
                Operation(
                    type=OperationType.FORWARD,
                    data=create_batch_data(["lora_a", "lora_b"]),
                ),
                Operation(
                    type=OperationType.UNLOAD,
                    data="lora_b",
                ),
                Operation(
                    type=OperationType.FORWARD,
                    data=create_batch_data("lora_a"),
                ),
            ],
        )
        self._run_test_case(test_case)

    def test_npu_lora_load_without_initial_adapters(self):
        """Test loading LoRA adapters without initial adapters."""
        test_case = TestCase(
            description="Dynamic LoRA update without initial lora_paths on NPU",
            base=LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            max_loras_per_batch=2,
            all_adapters=[self.lora_a, self.lora_b],
            enable_lora=True,
            max_lora_rank=64,
            lora_target_modules=["all"],
            op_sequence=[
                Operation(
                    type=OperationType.LOAD,
                    data="lora_a",
                ),
                Operation(
                    type=OperationType.LOAD,
                    data="lora_b",
                ),
                Operation(
                    type=OperationType.FORWARD,
                    data=create_batch_data(["lora_a", "lora_b"]),
                ),
                Operation(
                    type=OperationType.UNLOAD,
                    data="lora_a",
                ),
                Operation(
                    type=OperationType.FORWARD,
                    data=create_batch_data("lora_a"),
                ),
            ],
        )
        self._run_test_case(test_case)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
