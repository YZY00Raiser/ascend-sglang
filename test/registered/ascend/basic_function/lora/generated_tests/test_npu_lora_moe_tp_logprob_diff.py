"""
Test NPU LoRA MoE Tensor Parallel logprob diff - 验证MoE LoRA在TP=1和TP=2下的一致性

Tests MoE LoRA output consistency between TP=1 and TP=2 on Ascend NPU:
1. Logprob difference validation for MoE models with LoRA
2. TP=1 vs TP=2 output consistency verification
3. Threshold validation for logprob differences
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

# Test configuration - Using a smaller MoE model for testing
MODEL_NAME = "Qwen/Qwen1.5-MoE-A2.7B-Chat"  # Smaller MoE model for testing
ADAPTER_URL = "https://huggingface.co/sglang/qwen-moe-lora-test"

PROMPTS = [
    "Explain the concept of mixture of experts in machine learning.",
    "What are the advantages of MoE architectures?",
    "Describe how expert routing works in MoE models.",
]

# Thresholds for logprob comparison
LOGPROB_DIFF_THRESHOLD = 0.05  # Maximum allowed logprob difference
TOP_K_TOKENS = 50  # Number of top tokens to compare


class TestNpuLoRAMoETPLogprobDiff:
    """Test MoE LoRA TP consistency on Ascend NPU."""

    @pytest.fixture(scope="class")
    def tp1_server(self):
        """Start TP=1 server for MoE LoRA testing."""
        print("\n[SGLang] Starting TP=1 server for MoE LoRA...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'moe-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "moe_runner_backend": "ascend",  # Use Ascend MoE backend
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping TP=1 server...")
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def tp2_server(self):
        """Start TP=2 server for MoE LoRA testing."""
        print("\n[SGLang] Starting TP=2 server for MoE LoRA...")
        
        # Check available NPUs
        npu_count = get_available_npu_count()
        if npu_count < 2:
            pytest.skip(f"TP=2 test requires 2 NPUs, but only {npu_count} available")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 2,
            "lora_paths": f"{{'moe-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "moe_runner_backend": "ascend",
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping TP=2 server...")
        kill_sglang_server(server_process)

    def _get_logprobs(self, prompt: str, lora_path: str = None, server_url: str = None) -> Dict:
        """Get logprobs for a prompt."""
        response = call_sglang_generate(
            prompt=prompt,
            lora_path=lora_path,
            max_tokens=1,
            temperature=0.0,
            logprobs=TOP_K_TOKENS,
            server_url=server_url,
        )
        
        # Extract logprobs
        logprobs_dict = {}
        if "logprobs" in response and "content" in response["logprobs"]:
            for token_info in response["logprobs"]["content"]:
                if "top_logprobs" in token_info:
                    for item in token_info["top_logprobs"]:
                        token = item.get("token", "")
                        logprob = item.get("logprob", 0.0)
                        logprobs_dict[token] = logprob
        
        return logprobs_dict

    def test_moe_lora_tp1_tp2_consistency(self, tp1_server, tp2_server):
        """Test MoE LoRA output consistency between TP=1 and TP=2."""
        print("\n[Test] Testing MoE LoRA TP=1 vs TP=2 consistency...")
        
        max_logprob_diff = 0.0
        all_differences = []
        
        for prompt in PROMPTS:
            print(f"\n  Prompt: {prompt[:50]}...")
            
            # Get logprobs from TP=1
            logprobs_tp1 = self._get_logprobs(
                prompt, 
                lora_path="moe-lora",
                server_url="http://localhost:30000"
            )
            
            # Get logprobs from TP=2
            logprobs_tp2 = self._get_logprobs(
                prompt,
                lora_path="moe-lora", 
                server_url="http://localhost:30001"
            )
            
            # Compare logprobs for common tokens
            common_tokens = set(logprobs_tp1.keys()) & set(logprobs_tp2.keys())
            
            if not common_tokens:
                print("    Warning: No common tokens found")
                continue
            
            differences = []
            for token in common_tokens:
                lp1 = logprobs_tp1[token]
                lp2 = logprobs_tp2[token]
                diff = abs(lp1 - lp2)
                differences.append(diff)
                max_logprob_diff = max(max_logprob_diff, diff)
                
                if diff > LOGPROB_DIFF_THRESHOLD:
                    print(f"    Token '{token}': TP1={lp1:.4f}, TP2={lp2:.4f}, diff={diff:.4f}")
            
            avg_diff = sum(differences) / len(differences)
            all_differences.extend(differences)
            print(f"    Average diff: {avg_diff:.6f}, Max diff: {max(differences):.6f}")
        
        # Summary
        print(f"\n[Summary]")
        print(f"  Total token comparisons: {len(all_differences)}")
        print(f"  Maximum logprob difference: {max_logprob_diff:.6f}")
        print(f"  Average logprob difference: {sum(all_differences)/len(all_differences):.6f}")
        print(f"  Threshold: {LOGPROB_DIFF_THRESHOLD}")
        
        # Assert that differences are within threshold
        assert max_logprob_diff <= LOGPROB_DIFF_THRESHOLD, (
            f"Maximum logprob difference ({max_logprob_diff:.6f}) exceeds threshold "
            f"({LOGPROB_DIFF_THRESHOLD})"
        )
        
        print("  ✅ MoE LoRA TP=1 and TP=2 outputs are consistent")

    def test_moe_lora_basic_inference(self, tp1_server):
        """Test basic MoE LoRA inference on Ascend NPU."""
        print("\n[Test] Testing basic MoE LoRA inference...")
        
        for prompt in PROMPTS[:2]:
            print(f"  Prompt: {prompt[:40]}...")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="moe-lora",
                max_tokens=100,
                temperature=0.7,
                server_url="http://localhost:30000",
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            print(f"    Generated: {text[:60]}...")
            
            assert text and len(text) > 0, "Should generate non-empty text"
        
        print("  ✅ Basic MoE LoRA inference works correctly")

    def test_moe_lora_generation_quality(self, tp1_server):
        """Test MoE LoRA generation quality with ROUGE-L."""
        print("\n[Test] Testing MoE LoRA generation quality...")
        
        # Generate completions
        prompt = "Explain the concept of mixture of experts."
        reference = "Mixture of Experts (MoE) is a machine learning technique that combines multiple expert models."
        
        response = call_sglang_generate(
            prompt=prompt,
            lora_path="moe-lora",
            max_tokens=100,
            temperature=0.0,
            server_url="http://localhost:30000",
        )
        
        generated = response.get("choices", [{}])[0].get("text", "")
        
        # Compute ROUGE-L (simplified)
        score = self._compute_rouge_l(reference, generated)
        print(f"  Generated: {generated[:60]}...")
        print(f"  ROUGE-L score: {score:.4f}")
        
        # Just check that score is reasonable (not too low)
        assert score > 0.05, f"ROUGE-L score too low: {score:.4f}"
        
        print("  ✅ Generation quality is acceptable")

    def _compute_rouge_l(self, reference: str, hypothesis: str) -> float:
        """Compute simplified ROUGE-L score."""
        def lcs_length(x: str, y: str) -> int:
            m, n = len(x), len(y)
            if m == 0 or n == 0:
                return 0
            
            prev = [0] * (n + 1)
            curr = [0] * (n + 1)
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        curr[j] = prev[j-1] + 1
                    else:
                        curr[j] = max(prev[j], curr[j-1])
                prev, curr = curr, prev
            
            return prev[n]
        
        lcs = lcs_length(reference, hypothesis)
        if lcs == 0:
            return 0.0
        
        precision = lcs / len(hypothesis) if len(hypothesis) > 0 else 0
        recall = lcs / len(reference) if len(reference) > 0 else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
