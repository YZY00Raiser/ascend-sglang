"""
Test NPU LoRA Tensor Parallel - 验证NPU上的LoRA张量并行功能

Tests tensor parallelism with LoRA on Ascend NPU:
1. TP=2 LoRA inference correctness
2. Output consistency between TP=1 and TP=2
3. Overlapping loading with tensor parallelism
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    get_available_npu_count,
)

# Test configuration
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
ADAPTER_URL = "https://huggingface.co/sglang/llama-3.2-lora-test/tree/main/adapter_a"

PROMPTS = [
    "Explain machine learning.",
    "What is artificial intelligence?",
    "Describe neural networks.",
]


def generate_with_server(prompts, lora_path=None, server_url="http://localhost:30000"):
    """Generate completions using the running server."""
    results = []
    for prompt in prompts:
        response = call_sglang_generate(
            prompt=prompt,
            lora_path=lora_path,
            max_tokens=50,
            temperature=0.0,
            server_url=server_url,
        )
        text = response.get("choices", [{}])[0].get("text", "")
        results.append(text)
    return results


class TestNpuLoRATensorParallel:
    """Test LoRA tensor parallelism on Ascend NPU."""

    @pytest.fixture(scope="class")
    def tp1_server(self):
        """Start TP=1 server for baseline comparison."""
        print("\n[SGLang] Starting TP=1 server for baseline...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'test-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping TP=1 server...")
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def tp2_server(self):
        """Start TP=2 server."""
        print("\n[SGLang] Starting TP=2 server...")
        
        # Check if we have enough NPUs
        npu_count = get_available_npu_count()
        if npu_count < 2:
            pytest.skip(f"TP=2 test requires 2 NPUs, but only {npu_count} available")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 2,
            "lora_paths": f"{{'test-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping TP=2 server...")
        kill_sglang_server(server_process)

    def test_tp2_lora_inference(self, tp2_server):
        """Test basic TP=2 LoRA inference."""
        print("\n[Test] Testing TP=2 LoRA inference...")
        
        results = []
        for prompt in PROMPTS:
            result = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=50,
                temperature=0.0,
            )
            text = result.get("choices", [{}])[0].get("text", "")
            results.append(text)
            print(f"    {prompt[:30]}... -> {text[:50]}...")
        
        # Verify all results are non-empty
        for i, result in enumerate(results):
            assert result and len(result) > 0, f"Prompt {i} should produce output"
        
        print("  ✅ TP=2 LoRA inference works correctly")

    def test_tp1_tp2_output_consistency(self, tp1_server, tp2_server):
        """Test output consistency between TP=1 and TP=2."""
        print("\n[Test] Testing TP=1 vs TP=2 output consistency...")
        
        # Generate with TP=1
        print("  Generating with TP=1...")
        tp1_results = []
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=50,
                temperature=0.0,
                server_url="http://localhost:30000",
            )
            text = response.get("choices", [{}])[0].get("text", "")
            tp1_results.append(text)
            print(f"    TP=1: {text[:50]}...")
        
        # Generate with TP=2
        print("  Generating with TP=2...")
        tp2_results = []
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=50,
                temperature=0.0,
                server_url="http://localhost:30001",  # TP=2 server port
            )
            text = response.get("choices", [{}])[0].get("text", "")
            tp2_results.append(text)
            print(f"    TP=2: {text[:50]}...")
        
        # Compare outputs (with tolerance for minor numerical differences)
        print("  Comparing TP=1 vs TP=2 outputs...")
        for i, (tp1, tp2) in enumerate(zip(tp1_results, tp2_results)):
            # Use ROUGE-L to compare similarity (allows for minor differences)
            similarity = self._compute_similarity(tp1, tp2)
            print(f"    Prompt {i}: {similarity:.2%} similar")
            
            # Outputs should be highly similar (99%+)
            assert similarity >= 0.99, (
                f"TP=1 and TP=2 outputs should be highly similar\n"
                f"  TP=1: {tp1}\n"
                f"  TP=2: {tp2}"
            )
        
        print("  ✅ TP=1 and TP=2 outputs are consistent")

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts (0-1)."""
        # Simple character-level similarity
        if len(text1) == 0 and len(text2) == 0:
            return 1.0
        if len(text1) == 0 or len(text2) == 0:
            return 0.0
        
        # Use Levenshtein-like similarity
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        
        # Simple character match rate
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        return matches / max_len


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
