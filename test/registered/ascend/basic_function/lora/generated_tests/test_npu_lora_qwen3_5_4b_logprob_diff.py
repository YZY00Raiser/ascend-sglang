"""
Test NPU LoRA Qwen3.5-4B logprob diff - 验证Qwen3.5-4B模型LoRA精度

Tests Qwen3.5-4B LoRA precision on Ascend NPU:
1. Logprob comparison between reference implementation and SGLang
2. Small model LoRA precision validation
"""

import os
import sys
import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
)

# Test configuration
MODEL_NAME = "Qwen/Qwen3.5-4B"
LORA_ADAPTER = "sglang/qwen3.5-4b-lora-test"

PROMPTS = [
    "解释机器学习的基本原理。",
    "什么是神经网络？请简要说明。",
    "描述深度学习在计算机视觉中的应用。",
    "自然语言处理的主要任务有哪些？",
]

# Logprob comparison threshold
LOGPROB_DIFF_THRESHOLD = 0.01
TOP_K_TOKENS = 50


class TestNpuLoRAQwen35LogprobDiff:
    """Test Qwen3.5-4B LoRA precision on Ascend NPU."""

    @pytest.fixture(scope="class")
    def hf_results(self):
        """Compute reference results using HF Transformers + PEFT."""
        print("\n[HF] Loading Qwen3.5-4B base model and LoRA adapter...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        print("  Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        print("  Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, LORA_ADAPTER)
        model.eval()
        
        # Compute logprobs for each prompt
        results = {}
        for prompt in PROMPTS:
            print(f"  Processing: {prompt[:40]}...")
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Get logprobs for next token
                next_token_logits = logits[0, -1, :]
                logprobs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k
                topk_logprobs, topk_indices = torch.topk(logprobs, TOP_K_TOKENS)
                
                token_logprobs = {}
                for idx, logprob in zip(topk_indices.tolist(), topk_logprobs.tolist()):
                    token_str = tokenizer.decode([idx])
                    token_logprobs[token_str] = logprob
            
            results[prompt] = token_logprobs
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return results

    @pytest.fixture(scope="class")
    def sglang_server(self):
        """Start SGLang server with Qwen3.5-4B."""
        print("\n[SGLang] Starting server with Qwen3.5-4B...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'qwen35-lora':'{LORA_ADAPTER}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            "trust_remote_code": True,
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def sglang_results(self, sglang_server):
        """Compute SGLang results."""
        print("\n[SGLang] Computing logprobs...")
        
        results = {}
        for prompt in PROMPTS:
            print(f"  Processing: {prompt[:40]}...")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="qwen35-lora",
                max_tokens=1,
                temperature=0.0,
                logprobs=TOP_K_TOKENS,
            )
            
            logprobs_dict = {}
            if "logprobs" in response and "content" in response["logprobs"]:
                for token_info in response["logprobs"]["content"]:
                    if "top_logprobs" in token_info:
                        for item in token_info["top_logprobs"]:
                            token = item.get("token", "")
                            logprob = item.get("logprob", 0.0)
                            logprobs_dict[token] = logprob
            
            results[prompt] = logprobs_dict
        
        return results

    def test_logprob_equivalence(self, hf_results, sglang_results):
        """Test logprob equivalence between HF and SGLang."""
        print("\n[Test] Comparing HF and SGLang logprobs...")
        
        max_diff = 0.0
        all_diffs = []
        
        for prompt in PROMPTS:
            hf_logprobs = hf_results[prompt]
            sgl_logprobs = sglang_results[prompt]
            
            common_tokens = set(hf_logprobs.keys()) & set(sgl_logprobs.keys())
            
            if not common_tokens:
                print(f"  Warning: No common tokens for prompt: {prompt[:40]}...")
                continue
            
            for token in common_tokens:
                diff = abs(hf_logprobs[token] - sgl_logprobs[token])
                all_diffs.append(diff)
                max_diff = max(max_diff, diff)
                
                if diff > LOGPROB_DIFF_THRESHOLD:
                    print(f"    Token '{token}': HF={hf_logprobs[token]:.4f}, SGL={sgl_logprobs[token]:.4f}, diff={diff:.4f}")
        
        print(f"\n[Summary]")
        print(f"  Total comparisons: {len(all_diffs)}")
        print(f"  Maximum difference: {max_diff:.6f}")
        print(f"  Average difference: {sum(all_diffs)/len(all_diffs):.6f}")
        print(f"  Threshold: {LOGPROB_DIFF_THRESHOLD}")
        
        assert max_diff <= LOGPROB_DIFF_THRESHOLD, (
            f"Maximum logprob difference ({max_diff:.6f}) exceeds threshold "
            f"({LOGPROB_DIFF_THRESHOLD})"
        )
        
        print("  ✅ Logprob equivalence verified")

    def test_basic_generation(self, sglang_server):
        """Test basic generation with Qwen3.5-4B LoRA."""
        print("\n[Test] Testing basic Qwen3.5-4B LoRA generation...")
        
        for prompt in PROMPTS[:2]:
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="qwen35-lora",
                max_tokens=50,
                temperature=0.7,
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            print(f"    {prompt[:40]}... -> {text[:60]}...")
            
            assert text and len(text) > 0, "Should generate non-empty text"
        
        print("  ✅ Basic generation works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
