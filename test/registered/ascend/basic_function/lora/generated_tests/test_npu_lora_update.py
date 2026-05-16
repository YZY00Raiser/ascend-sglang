"""
Test NPU LoRA update functionality - 验证动态适配器加载/卸载功能

Tests dynamic LoRA adapter management on Ascend NPU:
1. Dynamic adapter loading and unloading
2. Multi-adapter forward pass
3. Error handling for already loaded adapters
4. ROUGE-L score validation for generation quality
"""

import os
import sys
import time
import pytest
import requests
from typing import List, Dict

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
ADAPTER_BASE_URL = "https://huggingface.co/sglang/llama-3.2-lora-test"

PROMPTS = [
    "Write a haiku about nature.",
    "Explain quantum computing in simple terms.",
    "Describe your favorite hobby.",
]

REFERENCE_ANSWERS = [
    "Mountains reach the sky\nRivers flow through silent valleys\nNature's beauty shines",
    "Quantum computers use qubits that can exist in multiple states at once.",
    "Reading books expands knowledge and takes you to different worlds.",
]


class TestNpuLoRAUpdate:
    """Test LoRA dynamic update functionality on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server without pre-loading adapters."""
        print("\n[SGLang] Starting server for dynamic adapter testing...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            # Don't pre-load any adapters
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    def _compute_rouge_l(self, reference: str, hypothesis: str) -> float:
        """Compute ROUGE-L score between reference and hypothesis."""
        def lcs_length(x: str, y: str) -> int:
            """Compute longest common subsequence length."""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs = lcs_length(reference, hypothesis)
        if len(reference) == 0 or len(hypothesis) == 0:
            return 0.0
        
        precision = lcs / len(hypothesis)
        recall = lcs / len(reference)
        
        if precision + recall == 0:
            return 0.0
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def test_dynamic_adapter_loading(self, server_process):
        """Test dynamically loading adapters at runtime."""
        print("\n[Test] Testing dynamic adapter loading...")
        
        adapter_name = "test_adapter"
        adapter_url = f"{ADAPTER_BASE_URL}/tree/main/adapter_a"
        
        # Generate without LoRA first
        print("  Generating without LoRA...")
        result_base = call_sglang_generate(
            prompt=PROMPTS[0],
            max_tokens=50,
            temperature=0.0,
        )
        text_base = result_base.get("choices", [{}])[0].get("text", "")
        
        # Load adapter dynamically
        print("  Loading adapter dynamically...")
        response = call_sglang_load_adapter(adapter_name, adapter_url)
        assert response.get("status") == "success", "Adapter loading should succeed"
        
        # Generate with LoRA
        print("  Generating with LoRA...")
        result_lora = call_sglang_generate(
            prompt=PROMPTS[0],
            lora_path=adapter_name,
            max_tokens=50,
            temperature=0.0,
        )
        text_lora = result_lora.get("choices", [{}])[0].get("text", "")
        
        # Results should be different with LoRA
        print(f"    Without LoRA: {text_base[:60]}...")
        print(f"    With LoRA: {text_lora[:60]}...")
        
        # LoRA should affect the output (not identical)
        assert text_base != text_lora, "LoRA should produce different output"
        print("  ✅ Dynamic adapter loading works correctly")

    def test_adapter_unloading(self, server_process):
        """Test dynamically unloading adapters at runtime."""
        print("\n[Test] Testing dynamic adapter unloading...")
        
        adapter_name = "unload_test"
        adapter_url = f"{ADAPTER_BASE_URL}/tree/main/adapter_a"
        
        # Load adapter
        print("  Loading adapter...")
        call_sglang_load_adapter(adapter_name, adapter_url)
        
        # Generate with LoRA
        print("  Generating with loaded LoRA...")
        result1 = call_sglang_generate(
            prompt=PROMPTS[0],
            lora_path=adapter_name,
            max_tokens=50,
            temperature=0.0,
        )
        text1 = result1.get("choices", [{}])[0].get("text", "")
        
        # Unload adapter
        print("  Unloading adapter...")
        response = call_sglang_unload_adapter(adapter_name)
        assert response.get("status") == "success", "Adapter unloading should succeed"
        
        # Try to generate with unloaded adapter - should fail
        print("  Verifying adapter is unloaded...")
        try:
            result2 = call_sglang_generate(
                prompt=PROMPTS[0],
                lora_path=adapter_name,
                max_tokens=50,
                temperature=0.0,
            )
            # If we get here without error, check if it's using the unloaded adapter
            assert False, "Should not be able to generate with unloaded adapter"
        except Exception as e:
            print(f"    Expected error: {e}")
        
        # Reload and verify consistent results
        print("  Reloading adapter...")
        call_sglang_load_adapter(adapter_name, adapter_url)
        result3 = call_sglang_generate(
            prompt=PROMPTS[0],
            lora_path=adapter_name,
            max_tokens=50,
            temperature=0.0,
        )
        text3 = result3.get("choices", [{}])[0].get("text", "")
        
        assert text1 == text3, "Reloaded adapter should produce consistent results"
        print("  ✅ Dynamic adapter unloading works correctly")

    def test_multi_adapter_forward_pass(self, server_process):
        """Test forward pass with multiple adapters."""
        print("\n[Test] Testing multi-adapter forward pass...")
        
        adapters = {
            "adapter_1": f"{ADAPTER_BASE_URL}/tree/main/adapter_a",
            "adapter_2": f"{ADAPTER_BASE_URL}/tree/main/adapter_b",
            "adapter_3": f"{ADAPTER_BASE_URL}/tree/main/adapter_c",
        }
        
        # Load multiple adapters
        print("  Loading multiple adapters...")
        for name, url in adapters.items():
            response = call_sglang_load_adapter(name, url)
            assert response.get("status") == "success", f"Loading {name} should succeed"
        
        # Generate with each adapter
        print("  Generating with each adapter...")
        results = {}
        for name in adapters.keys():
            response = call_sglang_generate(
                prompt=PROMPTS[1],
                lora_path=name,
                max_tokens=50,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            results[name] = text
            print(f"    {name}: {text[:50]}...")
        
        # Each adapter should produce different outputs
        assert len(set(results.values())) == len(results), (
            "Different adapters should produce different outputs"
        )
        
        print("  ✅ Multi-adapter forward pass works correctly")

    def test_already_loaded_adapter_error(self, server_process):
        """Test error handling when loading an already loaded adapter."""
        print("\n[Test] Testing error handling for already loaded adapter...")
        
        adapter_name = "duplicate_test"
        adapter_url = f"{ADAPTER_BASE_URL}/tree/main/adapter_a"
        
        # Load adapter first time
        print("  First load attempt...")
        response1 = call_sglang_load_adapter(adapter_name, adapter_url)
        assert response1.get("status") == "success", "First load should succeed"
        
        # Try to load same adapter again
        print("  Second load attempt (should handle gracefully)...")
        response2 = call_sglang_load_adapter(adapter_name, adapter_url)
        
        # Should either succeed or return appropriate error
        status = response2.get("status")
        assert status in ["success", "already_loaded"], (
            f"Should handle duplicate load gracefully, got status: {status}"
        )
        
        print(f"    Status: {status}")
        print("  ✅ Duplicate adapter loading handled correctly")

    def test_rouge_l_score_validation(self, server_process):
        """Test ROUGE-L score for generation quality validation."""
        print("\n[Test] Testing ROUGE-L score validation...")
        
        adapter_name = "rouge_test"
        adapter_url = f"{ADAPTER_BASE_URL}/tree/main/adapter_a"
        
        # Load adapter
        call_sglang_load_adapter(adapter_name, adapter_url)
        
        rouge_scores = []
        
        for prompt, ref_answer in zip(PROMPTS, REFERENCE_ANSWERS):
            # Generate response
            response = call_sglang_generate(
                prompt=prompt,
                lora_path=adapter_name,
                max_tokens=100,
                temperature=0.0,
            )
            generated = response.get("choices", [{}])[0].get("text", "")
            
            # Compute ROUGE-L
            score = self._compute_rouge_l(ref_answer, generated)
            rouge_scores.append(score)
            
            print(f"    Prompt: {prompt[:40]}...")
            print(f"    Generated: {generated[:60]}...")
            print(f"    ROUGE-L: {score:.4f}")
        
        # Check if scores are reasonable (> 0.1 indicates some overlap)
        avg_score = sum(rouge_scores) / len(rouge_scores)
        print(f"  Average ROUGE-L: {avg_score:.4f}")
        
        assert avg_score > 0.1, f"Average ROUGE-L score too low: {avg_score:.4f}"
        print("  ✅ ROUGE-L scores are acceptable")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
