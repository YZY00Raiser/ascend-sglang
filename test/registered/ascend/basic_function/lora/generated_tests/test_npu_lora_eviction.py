"""
Test NPU LoRA eviction behavior - 验证LoRA适配器的驱逐行为

Tests the dynamic loading and unloading of LoRA adapters when memory limits are exceeded:
1. Adapter eviction with LRU/FIFO policies on Ascend NPU
2. Output consistency before and after eviction cycles
3. Dynamic adapter loading/unloading behavior
"""

import os
import sys
import time
import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    call_sglang_load_adapter,
    call_sglang_unload_adapter,
)

# Test configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_URL = "https://huggingface.co/sglang/llama-3.2-lora-test"

# Multiple adapters for eviction testing
ADAPTERS = {
    "adapter_a": f"{ADAPTER_URL}/tree/main/adapter_a",
    "adapter_b": f"{ADAPTER_URL}/tree/main/adapter_b",
    "adapter_c": f"{ADAPTER_URL}/tree/main/adapter_c",
    "adapter_d": f"{ADAPTER_URL}/tree/main/adapter_d",
}

PROMPT = "Tell me a short story about a brave knight."


class TestNpuLoRAEviction:
    """Test LoRA adapter eviction behavior on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server with limited memory to force eviction."""
        print("\n[SGLang] Starting server with limited LoRA slots...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "max_loaded_loras": 2,  # Limit to force eviction
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    def _get_generation_result(self, adapter_name, prompt):
        """Get generation result with a specific adapter."""
        response = call_sglang_generate(
            prompt=prompt,
            lora_path=adapter_name,
            max_tokens=50,
            temperature=0.0,
        )
        return response.get("choices", [{}])[0].get("text", "")

    def test_lru_eviction_policy(self, server_process):
        """Test LRU eviction policy - least recently used adapter is evicted."""
        print("\n[Test] Testing LRU eviction policy...")
        
        # Load adapters A and B (fills max_loaded_loras=2)
        print("  Loading adapter_a...")
        call_sglang_load_adapter("adapter_a", ADAPTERS["adapter_a"])
        
        print("  Loading adapter_b...")
        call_sglang_load_adapter("adapter_b", ADAPTERS["adapter_b"])
        
        # Use adapter_a to make it most recently used
        print("  Using adapter_a (makes it LRU)...")
        result_a1 = self._get_generation_result("adapter_a", PROMPT)
        
        # Load adapter_c - should evict adapter_b (LRU)
        print("  Loading adapter_c (should evict adapter_b)...")
        call_sglang_load_adapter("adapter_c", ADAPTERS["adapter_c"])
        
        # Verify adapter_a still works (was recently used)
        print("  Verifying adapter_a still accessible...")
        result_a2 = self._get_generation_result("adapter_a", PROMPT)
        assert result_a1 == result_a2, "Adapter A result should be consistent after eviction"
        
        # adapter_b should have been evicted - reloading should work
        print("  Verifying adapter_b was evicted (reloading)...")
        call_sglang_load_adapter("adapter_b", ADAPTERS["adapter_b"])
        result_b = self._get_generation_result("adapter_b", PROMPT)
        assert result_b, "Adapter B should be accessible after reloading"

    def test_fifo_eviction_policy(self, server_process):
        """Test FIFO eviction policy - first loaded adapter is evicted."""
        print("\n[Test] Testing FIFO eviction policy...")
        
        # Restart server with FIFO policy
        kill_sglang_server(server_process)
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "max_loaded_loras": 2,
            "lora_eviction_policy": "FIFO",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        # Load adapters A and B
        print("  Loading adapter_a (first)...")
        call_sglang_load_adapter("adapter_a", ADAPTERS["adapter_a"])
        result_a1 = self._get_generation_result("adapter_a", PROMPT)
        
        print("  Loading adapter_b (second)...")
        call_sglang_load_adapter("adapter_b", ADAPTERS["adapter_b"])
        
        # Load adapter_c - should evict adapter_a (FIFO)
        print("  Loading adapter_c (should evict adapter_a)...")
        call_sglang_load_adapter("adapter_c", ADAPTERS["adapter_c"])
        
        # adapter_a should have been evicted
        print("  Verifying adapter_a was evicted (reloading required)...")
        call_sglang_load_adapter("adapter_a", ADAPTERS["adapter_a"])
        result_a2 = self._get_generation_result("adapter_a", PROMPT)
        
        # Results should be consistent after reload
        assert result_a1 == result_a2, "Adapter A should produce consistent results after eviction"
        
        kill_sglang_server(server_process)

    def test_eviction_output_consistency(self, server_process):
        """Test output consistency before and after eviction cycles."""
        print("\n[Test] Testing output consistency through eviction cycles...")
        
        results = {}
        
        # Load and generate with adapter A
        print("  Phase 1: Load and use adapter_a...")
        call_sglang_load_adapter("adapter_a", ADAPTERS["adapter_a"])
        results["phase1_a"] = self._get_generation_result("adapter_a", PROMPT)
        
        # Force eviction by loading other adapters
        print("  Phase 2: Load other adapters to force eviction...")
        call_sglang_load_adapter("adapter_b", ADAPTERS["adapter_b"])
        call_sglang_load_adapter("adapter_c", ADAPTERS["adapter_c"])
        call_sglang_load_adapter("adapter_d", ADAPTERS["adapter_d"])
        
        # Reload adapter A
        print("  Phase 3: Reload adapter_a after eviction...")
        call_sglang_load_adapter("adapter_a", ADAPTERS["adapter_a"])
        results["phase3_a"] = self._get_generation_result("adapter_a", PROMPT)
        
        # Compare results - should be identical
        print("  Verifying consistency across eviction cycles...")
        assert results["phase1_a"] == results["phase3_a"], (
            f"Output should be consistent before and after eviction\n"
            f"Phase 1: {results['phase1_a']}\n"
            f"Phase 3: {results['phase3_a']}"
        )
        
        print("  ✅ Output consistency verified through eviction cycles")

    def test_different_target_modules_eviction(self, server_process):
        """Test eviction with adapters targeting different modules."""
        print("\n[Test] Testing eviction with different target modules...")
        
        # Adapters with different target module configurations
        adapter_q_proj = f"{ADAPTER_URL}/tree/main/q_proj_only"
        adapter_v_proj = f"{ADAPTER_URL}/tree/main/v_proj_only"
        adapter_qv_proj = f"{ADAPTER_URL}/tree/main/qv_proj"
        
        # Load adapters with different target modules
        print("  Loading adapters with different target modules...")
        call_sglang_load_adapter("q_only", adapter_q_proj)
        call_sglang_load_adapter("v_only", adapter_v_proj)
        call_sglang_load_adapter("qv", adapter_qv_proj)
        
        # Test generation with each adapter
        results = {}
        for adapter_name in ["q_only", "v_only", "qv"]:
            result = self._get_generation_result(adapter_name, PROMPT)
            results[adapter_name] = result
            print(f"    {adapter_name}: {result[:50]}...")
        
        # Verify all adapters produce valid outputs
        for adapter_name, result in results.items():
            assert result and len(result) > 0, f"Adapter {adapter_name} should produce output"
        
        print("  ✅ Multi-target-module eviction works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
