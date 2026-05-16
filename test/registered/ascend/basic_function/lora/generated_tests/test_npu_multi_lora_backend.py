"""
Test NPU Multi-LoRA backend - 验证多LoRA批次处理功能

Tests multi-LoRA batch processing on Ascend NPU:
1. Batch split equivalence - same results between batched and individual requests
2. Multiple adapters in same batch
3. Multiple batches with different adapters
"""

import os
import sys
import pytest
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    call_sglang_load_adapter,
    call_sglang_batch_generate,
)

# Test configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_BASE_URL = "https://huggingface.co/sglang/llama-3.2-lora-test"

PROMPTS = [
    "What is machine learning?",
    "Explain neural networks.",
    "Describe deep learning.",
    "What is natural language processing?",
    "Explain computer vision.",
]

ADAPTERS = {
    "adapter_a": f"{ADAPTER_BASE_URL}/tree/main/adapter_a",
    "adapter_b": f"{ADAPTER_BASE_URL}/tree/main/adapter_b",
    "adapter_c": f"{ADAPTER_BASE_URL}/tree/main/adapter_c",
}


class TestNpuMultiLoRABackend:
    """Test Multi-LoRA batch processing on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server for multi-LoRA testing."""
        print("\n[SGLang] Starting server for multi-LoRA backend testing...")
        
        # Load all adapters at startup
        lora_paths = {name: url.replace("/tree/main/", "/resolve/main/") 
                      for name, url in ADAPTERS.items()}
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": str(lora_paths),
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "max_loras_per_batch": 3,  # Allow up to 3 adapters per batch
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    def _generate_single(self, prompt: str, adapter: str = None) -> str:
        """Generate with a single request."""
        response = call_sglang_generate(
            prompt=prompt,
            lora_path=adapter,
            max_tokens=30,
            temperature=0.0,
        )
        return response.get("choices", [{}])[0].get("text", "")

    def _generate_batch(self, prompts: List[str], adapters: List[str] = None) -> List[str]:
        """Generate with batched requests."""
        requests = []
        for i, prompt in enumerate(prompts):
            req = {
                "prompt": prompt,
                "max_tokens": 30,
                "temperature": 0.0,
            }
            if adapters and i < len(adapters):
                req["lora_path"] = adapters[i]
            requests.append(req)
        
        responses = call_sglang_batch_generate(requests)
        return [r.get("choices", [{}])[0].get("text", "") for r in responses]

    def test_batch_split_equivalence_same_adapter(self, server_process):
        """Test that batched requests with same adapter produce same results as individual requests."""
        print("\n[Test] Testing batch split equivalence (same adapter)...")
        
        adapter = "adapter_a"
        test_prompts = PROMPTS[:3]
        
        # Generate individually
        print("  Generating individually...")
        individual_results = []
        for prompt in test_prompts:
            result = self._generate_single(prompt, adapter)
            individual_results.append(result)
            print(f"    Single: {result[:50]}...")
        
        # Generate as batch
        print("  Generating as batch...")
        batch_results = self._generate_batch(
            test_prompts, 
            [adapter] * len(test_prompts)
        )
        for i, result in enumerate(batch_results):
            print(f"    Batch:  {result[:50]}...")
        
        # Compare results
        for i, (ind, batch) in enumerate(zip(individual_results, batch_results)):
            assert ind == batch, (
                f"Results mismatch for prompt {i}:\n"
                f"  Individual: {ind}\n"
                f"  Batch:      {batch}"
            )
        
        print("  ✅ Batch split equivalence verified for same adapter")

    def test_batch_split_equivalence_no_adapter(self, server_process):
        """Test batch equivalence without LoRA adapters."""
        print("\n[Test] Testing batch split equivalence (no adapter)...")
        
        test_prompts = PROMPTS[:3]
        
        # Generate individually without LoRA
        print("  Generating individually without LoRA...")
        individual_results = []
        for prompt in test_prompts:
            result = self._generate_single(prompt, None)
            individual_results.append(result)
        
        # Generate as batch without LoRA
        print("  Generating as batch without LoRA...")
        batch_results = self._generate_batch(test_prompts, None)
        
        # Compare results
        for i, (ind, batch) in enumerate(zip(individual_results, batch_results)):
            assert ind == batch, (
                f"Results mismatch for prompt {i}:\n"
                f"  Individual: {ind}\n"
                f"  Batch:      {batch}"
            )
        
        print("  ✅ Batch split equivalence verified without adapter")

    def test_multiple_adapters_same_batch(self, server_process):
        """Test multiple different adapters in the same batch."""
        print("\n[Test] Testing multiple adapters in same batch...")
        
        # Each prompt with a different adapter
        test_configs = [
            ("What is Python?", "adapter_a"),
            ("What is JavaScript?", "adapter_b"),
            ("What is Rust?", "adapter_c"),
        ]
        
        prompts = [cfg[0] for cfg in test_configs]
        adapters = [cfg[1] for cfg in test_configs]
        
        # Generate as batch with mixed adapters
        print("  Generating batch with mixed adapters...")
        batch_results = self._generate_batch(prompts, adapters)
        
        # Generate individually for comparison
        print("  Generating individually for comparison...")
        individual_results = []
        for prompt, adapter in test_configs:
            result = self._generate_single(prompt, adapter)
            individual_results.append(result)
        
        # Compare results
        for i, (batch, ind) in enumerate(zip(batch_results, individual_results)):
            assert batch == ind, (
                f"Mismatch for adapter {adapters[i]}:\n"
                f"  Batch:      {batch}\n"
                f"  Individual: {ind}"
            )
        
        print("  ✅ Multiple adapters in same batch work correctly")

    def test_multiple_batches_different_adapters(self, server_process):
        """Test multiple batches with different adapter assignments."""
        print("\n[Test] Testing multiple batches with different adapters...")
        
        # Batch 1: All with adapter_a
        batch1_prompts = PROMPTS[:2]
        batch1_adapters = ["adapter_a"] * 2
        
        # Batch 2: All with adapter_b
        batch2_prompts = PROMPTS[2:4]
        batch2_adapters = ["adapter_b"] * 2
        
        # Batch 3: Mixed
        batch3_prompts = [PROMPTS[4]]
        batch3_adapters = ["adapter_c"]
        
        print("  Running batch 1 (adapter_a)...")
        results1 = self._generate_batch(batch1_prompts, batch1_adapters)
        
        print("  Running batch 2 (adapter_b)...")
        results2 = self._generate_batch(batch2_prompts, batch2_adapters)
        
        print("  Running batch 3 (adapter_c)...")
        results3 = self._generate_batch(batch3_prompts, batch3_adapters)
        
        # Verify all results
        all_results = results1 + results2 + results3
        assert len(all_results) == 5, "Should have 5 total results"
        
        for i, result in enumerate(all_results):
            assert result and len(result) > 0, f"Result {i} should not be empty"
        
        # Verify adaptars affected the outputs differently
        print("  Verifying different adapters produce different outputs...")
        
        # Use same prompt for different adapters to compare
        test_prompt = "Explain recursion."
        result_a = self._generate_single(test_prompt, "adapter_a")
        result_b = self._generate_single(test_prompt, "adapter_b")
        result_c = self._generate_single(test_prompt, "adapter_c")
        
        # Different adapters should produce different outputs
        assert result_a != result_b, "adapter_a and adapter_b should produce different outputs"
        assert result_b != result_c, "adapter_b and adapter_c should produce different outputs"
        
        print("  ✅ Multiple batches with different adapters work correctly")

    def test_large_batch_with_mixed_adapters(self, server_process):
        """Test large batch with mixed adapters to stress test batching logic."""
        print("\n[Test] Testing large batch with mixed adapters...")
        
        # Create a large batch with adapter mixing
        batch_size = 8
        prompts = [f"Question {i}: What is AI?" for i in range(batch_size)]
        
        # Cycle through adapters
        adapter_list = list(ADAPTERS.keys())
        adapters = [adapter_list[i % len(adapter_list)] for i in range(batch_size)]
        
        print(f"  Generating {batch_size} requests with mixed adapters...")
        batch_results = self._generate_batch(prompts, adapters)
        
        # Verify all results
        assert len(batch_results) == batch_size, f"Expected {batch_size} results"
        
        for i, result in enumerate(batch_results):
            assert result and len(result) > 0, f"Result {i} should not be empty"
            print(f"    [{adapters[i]}] {result[:40]}...")
        
        # Verify distinct outputs for different adapters (using same prompt)
        same_prompt = "Explain cloud computing."
        results_by_adapter = {}
        for adapter in adapter_list:
            results_by_adapter[adapter] = self._generate_single(same_prompt, adapter)
        
        # Check that different adapters produce different outputs
        unique_outputs = set(results_by_adapter.values())
        print(f"  Unique outputs from {len(adapter_list)} adapters: {len(unique_outputs)}")
        
        assert len(unique_outputs) > 1, "Different adapters should produce different outputs"
        
        print("  ✅ Large batch with mixed adapters works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
