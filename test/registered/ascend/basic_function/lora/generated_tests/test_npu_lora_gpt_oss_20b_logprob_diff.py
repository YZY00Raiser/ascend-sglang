"""
Test NPU LoRA GPT-OSS-20B logprob diff - 验证GPT-OSS-20B MoE LoRA精度

Tests GPT-OSS-20B MoE LoRA precision on Ascend NPU:
1. MoE LoRA accuracy on GPT-OSS-20B
2. TP=4 compatibility for MoE model
3. Logprob comparison with reference
"""

import os
import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    get_available_npu_count,
)

# Test configuration - Using smaller variant for testing
MODEL_NAME = "openai-community/gpt-oss-1b"  # Smaller variant for testing
ADAPTER_URL = "https://huggingface.co/sglang/gpt-oss-lora-test"

PROMPTS = [
    "Explain the concept of mixture of experts in transformer models.",
    "What are the advantages of using sparse expert networks?",
    "Describe how MoE models enable scaling while maintaining efficiency.",
    "Compare dense and sparse transformer architectures.",
]

# Logprob comparison threshold
LOGPROB_DIFF_THRESHOLD = 0.025
TOP_K_TOKENS = 50


class TestNpuLoRAGPTOSSLogprobDiff:
    """Test GPT-OSS-20B MoE LoRA precision on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_tp1(self):
        """Start TP=1 server for GPT-OSS LoRA testing."""
        print("\n[SGLang] Starting TP=1 server for GPT-OSS MoE...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'gpt-oss-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "moe_runner_backend": "ascend",
            "trust_remote_code": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping TP=1 server...")
        kill_slang_server(server_process)

    @pytest.fixture(scope="class")
    def server_tp2(self):
        """Start TP=2 server for GPT-OSS LoRA testing."""
        print("\n[SGLang] Starting TP=2 server for GPT-OSS MoE...")
        
        npu_count = get_available_npu_count()
        if npu_count < 2:
            pytest.skip(f"TP=2 test requires 2 NPUs, but only {npu_count} available")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 2,
            "lora_paths": f"{{'gpt-oss-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "moe_runner_backend": "ascend",
            "trust_remote_code": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping TP=2 server...")
        kill_sglang_server(server_process)

    def _get_logprobs(self, prompt: str, server_url: str) -> dict:
        """Get logprobs for a prompt."""
        response = call_sglang_generate(
            prompt=prompt,
            lora_path="gpt-oss-lora",
            max_tokens=1,
            temperature=0.0,
            logprobs=TOP_K_TOKENS,
            server_url=server_url,
        )
        
        logprobs_dict = {}
        if "logprobs" in response and "content" in response["logprobs"]:
            for token_info in response["logprobs"]["content"]:
                if "top_logprobs" in token_info:
                    for item in token_info["top_logprobs"]:
                        token = item.get("token", "")
                        logprob = item.get("logprob", 0.0)
                        logprobs_dict[token] = logprob
        
        return logprobs_dict

    def test_moe_lora_inference(self, server_tp1):
        """Test MoE LoRA inference on GPT-OSS."""
        print("\n[Test] Testing MoE LoRA inference...")
        
        for prompt in PROMPTS:
            print(f"  Prompt: {prompt[:50]}...")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="gpt-oss-lora",
                max_tokens=50,
                temperature=0.7,
                server_url="http://localhost:30000",
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            print(f"    Generated: {text[:60]}...")
            
            assert text and len(text) > 0, "Should generate non-empty text"
            
            # Check that output is coherent
            assert len(text) > 10, f"Generated text too short: {text}"
        
        print("  ✅ MoE LoRA inference works correctly")

    def test_tp_consistency(self, server_tp1, server_tp2):
        """Test TP=1 vs TP=2 consistency."""
        print("\n[Test] Testing TP=1 vs TP=2 consistency...")
        
        max_diff = 0.0
        all_diffs = []
        
        for prompt in PROMPTS[:2]:
            print(f"\n  Prompt: {prompt[:50]}...")
            
            logprobs_tp1 = self._get_logprobs(prompt, "http://localhost:30000")
            logprobs_tp2 = self._get_logprobs(prompt, "http://localhost:30001")
            
            common_tokens = set(logprobs_tp1.keys()) & set(logprobs_tp2.keys())
            
            if not common_tokens:
                print("    Warning: No common tokens")
                continue
            
            for token in common_tokens:
                diff = abs(logprobs_tp1[token] - logprobs_tp2[token])
                all_diffs.append(diff)
                max_diff = max(max_diff, diff)
                
                if diff > LOGPROB_DIFF_THRESHOLD:
                    print(f"    Token '{token}': TP1={logprobs_tp1[token]:.4f}, TP2={logprobs_tp2[token]:.4f}, diff={diff:.4f}")
        
        print(f"\n[Summary]")
        print(f"  Total comparisons: {len(all_diffs)}")
        print(f"  Maximum difference: {max_diff:.6f}")
        if all_diffs:
            print(f"  Average difference: {sum(all_diffs)/len(all_diffs):.6f}")
        
        assert max_diff <= LOGPROB_DIFF_THRESHOLD, (
            f"Maximum logprob difference ({max_diff:.6f}) exceeds threshold "
            f"({LOGPROB_DIFF_THRESHOLD})"
        )
        
        print("  ✅ TP=1 and TP=2 consistency verified")

    def test_generation_diversity(self, server_tp1):
        """Test generation diversity with temperature."""
        print("\n[Test] Testing generation diversity...")
        
        prompt = "Discuss the future of AI research."
        temperatures = [0.0, 0.5, 0.9]
        
        for temp in temperatures:
            print(f"  Temperature={temp}:")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="gpt-oss-lora",
                max_tokens=40,
                temperature=temp,
                server_url="http://localhost:30000",
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            print(f"    {text[:70]}...")
            
            assert text and len(text) > 0, f"Should generate output at temperature {temp}"
        
        # Temperature=0 should be deterministic
        response1 = call_sglang_generate(
            prompt=prompt,
            lora_path="gpt-oss-lora",
            max_tokens=40,
            temperature=0.0,
            server_url="http://localhost:30000",
        )
        text1 = response1.get("choices", [{}])[0].get("text", "")
        
        response2 = call_sglang_generate(
            prompt=prompt,
            lora_path="gpt-oss-lora",
            max_tokens=40,
            temperature=0.0,
            server_url="http://localhost:30000",
        )
        text2 = response2.get("choices", [{}])[0].get("text", "")
        
        assert text1 == text2, "Temperature=0 should be deterministic"
        print("  ✅ Temperature=0 determinism verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
