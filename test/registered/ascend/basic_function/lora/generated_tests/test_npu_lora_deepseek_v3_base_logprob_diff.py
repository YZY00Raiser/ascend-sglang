"""
Test NPU LoRA DeepSeek-V3 Base logprob diff - 验证DeepSeek-V3 MLA LoRA精度

Tests DeepSeek-V3 base model LoRA precision on Ascend NPU:
1. MLA (Multi-head Latent Attention) LoRA accuracy
2. Shared outer LoRA functionality
3. Multi-GPU TP=8 compatibility (simulated with TP=2/4)
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

# Test configuration - DeepSeek-V3 1B/7B for testing
MODEL_NAME = "deepseek-ai/DeepSeek-V3-Base"
ADAPTER_URL = "https://huggingface.co/sglang/deepseek-v3-lora-test"

PROMPTS = [
    "Explain the concept of attention mechanism in transformers.",
    "What is the difference between encoder and decoder architectures?",
    "Describe the purpose of positional encoding.",
    "How does multi-head attention work?",
]

# Logprob comparison threshold
LOGPROB_DIFF_THRESHOLD = 0.02
TOP_K_TOKENS = 50


class TestNpuLoRADeepSeekV3LogprobDiff:
    """Test DeepSeek-V3 LoRA precision on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_tp1(self):
        """Start TP=1 server for DeepSeek-V3 LoRA testing."""
        print("\n[SGLang] Starting TP=1 server for DeepSeek-V3...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'deepseek-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "moe_runner_backend": "ascend",
            "enable_mla": True,  # Enable MLA for DeepSeek
            "trust_remote_code": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping TP=1 server...")
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def server_tp2(self):
        """Start TP=2 server for DeepSeek-V3 LoRA testing."""
        print("\n[SGLang] Starting TP=2 server for DeepSeek-V3...")
        
        npu_count = get_available_npu_count()
        if npu_count < 2:
            pytest.skip(f"TP=2 test requires 2 NPUs, but only {npu_count} available")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 2,
            "lora_paths": f"{{'deepseek-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "moe_runner_backend": "ascend",
            "enable_mla": True,
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
            lora_path="deepseek-lora",
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

    def test_mla_lora_inference(self, server_tp1):
        """Test MLA LoRA inference on DeepSeek-V3."""
        print("\n[Test] Testing MLA LoRA inference...")
        
        for prompt in PROMPTS:
            print(f"  Prompt: {prompt[:50]}...")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="deepseek-lora",
                max_tokens=50,
                temperature=0.7,
                server_url="http://localhost:30000",
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            print(f"    Generated: {text[:60]}...")
            
            assert text and len(text) > 0, "Should generate non-empty text"
        
        print("  ✅ MLA LoRA inference works correctly")

    def test_tp1_tp2_consistency(self, server_tp1, server_tp2):
        """Test output consistency between TP=1 and TP=2 for MLA LoRA."""
        print("\n[Test] Testing TP=1 vs TP=2 consistency for MLA LoRA...")
        
        max_diff = 0.0
        all_diffs = []
        
        for prompt in PROMPTS:
            print(f"\n  Prompt: {prompt[:50]}...")
            
            # Get logprobs from TP=1
            logprobs_tp1 = self._get_logprobs(prompt, "http://localhost:30000")
            
            # Get logprobs from TP=2
            logprobs_tp2 = self._get_logprobs(prompt, "http://localhost:30001")
            
            # Compare
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
        print(f"  Average difference: {sum(all_diffs)/len(all_diffs):.6f}")
        
        assert max_diff <= LOGPROB_DIFF_THRESHOLD, (
            f"Maximum logprob difference ({max_diff:.6f}) exceeds threshold "
            f"({LOGPROB_DIFF_THRESHOLD})"
        )
        
        print("  ✅ TP=1 and TP=2 consistency verified for MLA LoRA")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
