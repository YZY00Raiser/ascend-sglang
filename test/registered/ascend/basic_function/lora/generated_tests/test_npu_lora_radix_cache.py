"""
Test NPU LoRA with Radix Cache - 验证NPU上LoRA与Radix缓存的配合

Tests LoRA interaction with Radix Cache on Ascend NPU:
1. LoRA with Radix Cache enabled/disabled
2. Cache hit/miss behavior with LoRA adapters
3. Prompt reuse efficiency with LoRA
"""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    call_sglang_get_server_info,
)

# Test configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_URL = "https://huggingface.co/sglang/llama-3.2-lora-test/tree/main/adapter_a"

PROMPTS = [
    "Explain the concept of machine learning.",
    "What is deep learning? Provide a detailed explanation.",
    "Describe artificial intelligence and its applications.",
]

# Same prompts to test cache reuse
REUSE_PROMPTS = [
    "Explain the concept of machine learning.",
    "Explain the concept of machine learning.",  # Same as above
    "What is deep learning? Provide a detailed explanation.",
    "What is deep learning? Provide a detailed explanation.",  # Same as above
]


class TestNpuLoRARadixCache:
    """Test LoRA with Radix Cache on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_with_cache(self):
        """Start SGLang server with Radix Cache enabled."""
        print("\n[SGLang] Starting server with Radix Cache enabled...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'test-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            # Radix Cache is enabled by default
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def server_without_cache(self):
        """Start SGLang server with Radix Cache disabled."""
        print("\n[SGLang] Starting server with Radix Cache disabled...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'test-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "disable_radix_cache": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    def test_lora_with_radix_cache_enabled(self, server_with_cache):
        """Test LoRA with Radix Cache enabled."""
        print("\n[Test] Testing LoRA with Radix Cache enabled...")
        
        results = []
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=50,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            results.append(text)
            print(f"    {prompt[:40]}... -> {text[:50]}...")
            
            assert text and len(text) > 0, "Should generate non-empty text"
        
        print("  ✅ LoRA with Radix Cache enabled works correctly")

    def test_lora_with_radix_cache_disabled(self, server_without_cache):
        """Test LoRA with Radix Cache disabled."""
        print("\n[Test] Testing LoRA with Radix Cache disabled...")
        
        results = []
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=50,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            results.append(text)
            print(f"    {prompt[:40]}... -> {text[:50]}...")
            
            assert text and len(text) > 0, "Should generate non-empty text"
        
        print("  ✅ LoRA with Radix Cache disabled works correctly")

    def test_radix_cache_reuse_with_lora(self, server_with_cache):
        """Test Radix Cache reuse with LoRA (same prompt)."""
        print("\n[Test] Testing Radix Cache reuse with LoRA...")
        
        # First pass - populate cache
        print("  First pass (populating cache)...")
        start_time = time.time()
        results_first = []
        for prompt in REUSE_PROMPTS[:2]:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=30,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            results_first.append(text)
        first_pass_time = time.time() - start_time
        print(f"    First pass time: {first_pass_time:.2f}s")
        
        # Second pass - should use cache
        print("  Second pass (using cache)...")
        start_time = time.time()
        results_second = []
        for prompt in REUSE_PROMPTS[:2]:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=30,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            results_second.append(text)
        second_pass_time = time.time() - start_time
        print(f"    Second pass time: {second_pass_time:.2f}s")
        
        # Results should be the same
        assert results_first == results_second, "Same prompt should produce same results"
        
        # Second pass should be faster (or similar) due to cache
        print(f"  Cache speedup: {first_pass_time/max(second_pass_time, 0.01):.2f}x")
        
        print("  ✅ Radix Cache reuse with LoRA works correctly")

    def test_different_adapters_same_prompt(self, server_with_cache):
        """Test same prompt with different LoRA adapters."""
        print("\n[Test] Testing same prompt with different adapters...")
        
        prompt = "Describe the future of technology."
        
        # First with LoRA
        response1 = call_sglang_generate(
            prompt=prompt,
            lora_path="test-lora",
            max_tokens=50,
            temperature=0.0,
        )
        text_with_lora = response1.get("choices", [{}])[0].get("text", "")
        
        # Then without LoRA
        response2 = call_sglang_generate(
            prompt=prompt,
            lora_path=None,
            max_tokens=50,
            temperature=0.0,
        )
        text_without_lora = response2.get("choices", [{}])[0].get("text", "")
        
        print(f"  With LoRA:    {text_with_lora[:60]}...")
        print(f"  Without LoRA: {text_without_lora[:60]}...")
        
        # Results should be different
        assert text_with_lora != text_without_lora, \
            "Same prompt with/without LoRA should produce different results"
        
        print("  ✅ Different adapters on same prompt work correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
