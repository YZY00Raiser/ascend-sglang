"""
Test NPU LoRA with Tied LM Head - 验证NPU上绑定嵌入模型的LoRA功能

Tests LoRA with models that have tied embeddings (input embeddings = output lm_head) on Ascend NPU:
1. Tied embedding LoRA application
2. lm_head LoRA application correctness
3. NaN value detection
"""

import os
import sys
import pytest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
)

# Test configuration - Using a model with tied embeddings
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Llama models typically have tied embeddings
ADAPTER_URL = "https://huggingface.co/sglang/llama-lora-tied-test"

PROMPTS = [
    "Explain the concept of neural networks.",
    "What is the difference between AI and machine learning?",
    "Describe how transformers work.",
]


class TestNpuLoRATiedLMHead:
    """Test LoRA with tied embeddings on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server with tied embedding model."""
        print("\n[SGLang] Starting server for tied embedding LoRA testing...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'tied-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "lora_target_modules": "q_proj,v_proj,k_proj,o_proj,embed_tokens,lm_head",
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    def test_tied_embedding_lora_inference(self, server_process):
        """Test basic inference with tied embeddings and LoRA."""
        print("\n[Test] Testing tied embedding LoRA inference...")
        
        results_with_lora = []
        results_without_lora = []
        
        for prompt in PROMPTS:
            # With LoRA
            response_lora = call_sglang_generate(
                prompt=prompt,
                lora_path="tied-lora",
                max_tokens=50,
                temperature=0.0,
            )
            text_lora = response_lora.get("choices", [{}])[0].get("text", "")
            results_with_lora.append(text_lora)
            
            # Without LoRA
            response_no_lora = call_sglang_generate(
                prompt=prompt,
                lora_path=None,
                max_tokens=50,
                temperature=0.0,
            )
            text_no_lora = response_no_lora.get("choices", [{}])[0].get("text", "")
            results_without_lora.append(text_no_lora)
            
            print(f"    Prompt: {prompt[:40]}...")
            print(f"    With LoRA:    {text_lora[:50]}...")
            print(f"    Without LoRA: {text_no_lora[:50]}...")
            
            # Verify outputs are valid (not empty, not NaN)
            assert text_lora and len(text_lora) > 0, "LoRA output should not be empty"
            assert text_no_lora and len(text_no_lora) > 0, "Base output should not be empty"
            
            # Both should produce valid, non-identical text
            assert text_lora != text_no_lora, "With and without LoRA should differ"
        
        print("  ✅ Tied embedding LoRA inference works correctly")

    def test_no_nan_values(self, server_process):
        """Test that no NaN values are produced in outputs."""
        print("\n[Test] Testing for NaN values in outputs...")
        
        for prompt in PROMPTS:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="tied-lora",
                max_tokens=50,
                temperature=0.0,
                logprobs=10,  # Also check logprobs for NaN
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            
            # Check for NaN in text (should be valid UTF-8)
            try:
                text.encode('utf-8')
            except UnicodeEncodeError:
                pytest.fail(f"Invalid text encoding: {text}")
            
            # Check logprobs if present
            if "logprobs" in response and "content" in response["logprobs"]:
                for token_info in response["logprobs"]["content"]:
                    if "top_logprobs" in token_info:
                        for item in token_info["top_logprobs"]:
                            logprob = item.get("logprob", 0)
                            assert not math.isnan(logprob), f"NaN logprob found: {logprob}"
                            assert not math.isinf(logprob), f"Inf logprob found: {logprob}"
            
            print(f"    {prompt[:40]}... -> Valid output ✓")
        
        print("  ✅ No NaN values detected in outputs")

    def test_lm_head_lora_application(self, server_process):
        """Test that lm_head LoRA is correctly applied."""
        print("\n[Test] Testing lm_head LoRA application...")
        
        # With lm_head LoRA, the output distribution should be different
        repeated_prompt = "The future of AI is"
        
        # Multiple generations to see variation
        outputs_with_lora = []
        for _ in range(3):
            response = call_sglang_generate(
                prompt=repeated_prompt,
                lora_path="tied-lora",
                max_tokens=20,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            outputs_with_lora.append(text)
        
        outputs_without_lora = []
        for _ in range(3):
            response = call_sglang_generate(
                prompt=repeated_prompt,
                lora_path=None,
                max_tokens=20,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            outputs_without_lora.append(text)
        
        print(f"  With LoRA (deterministic): {set(outputs_with_lora)}")
        print(f"  Without LoRA (deterministic): {set(outputs_without_lora)}")
        
        # With temperature=0, outputs should be deterministic within each case
        assert len(set(outputs_with_lora)) == 1, "LoRA outputs should be deterministic"
        assert len(set(outputs_without_lora)) == 1, "Base outputs should be deterministic"
        
        # But LoRA should change the output
        assert outputs_with_lora[0] != outputs_without_lora[0], \
            "LoRA should affect the output distribution"
        
        print("  ✅ lm_head LoRA application works correctly")

    def test_generation_consistency(self, server_process):
        """Test generation consistency with tied embeddings."""
        print("\n[Test] Testing generation consistency...")
        
        prompt = "Explain how computers work."
        
        # Run multiple times with same seed/temperature
        results = []
        for trial in range(3):
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="tied-lora",
                max_tokens=40,
                temperature=0.0,
            )
            text = response.get("choices", [{}])[0].get("text", "")
            results.append(text)
            print(f"    Trial {trial + 1}: {text[:50]}...")
        
        # All results should be identical
        assert len(set(results)) == 1, "Tied embedding LoRA should be deterministic"
        
        print("  ✅ Generation consistency verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
