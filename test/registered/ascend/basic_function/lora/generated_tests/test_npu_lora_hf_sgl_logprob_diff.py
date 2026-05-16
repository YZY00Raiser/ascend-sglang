"""
Test NPU LoRA HF vs SGLang logprob diff - 验证HF+LoRA与SGLang+LoRA的对数概率一致性

Tests the numerical equivalence between:
1. HuggingFace Transformers + LoRA (reference implementation)
2. SGLang with LoRA support on Ascend NPU

This ensures SGLang's LoRA implementation produces consistent results with the reference.
"""

import os
import sys
import pytest
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Add sglang to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_utils import (
    is_in_ci,
    run_bash_command,
    DEFAULT_PROMPTS_FOR_TEST,
)
from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    HF_LORA_MODELS,
    SGLANG_LORA_PATHS,
)

# Test configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # Base model
LORA_ADAPTER = "tloen/alpaca-lora-7b"    # LoRA adapter
PROMPTS = DEFAULT_PROMPTS_FOR_TEST[:5]   # Use first 5 prompts for comparison

# Logprob comparison threshold
LOGPROB_DIFF_THRESHOLD = 0.01  # Maximum allowed difference in logprobs
TOP_K = 50  # Number of top tokens to compare


class TestNpuLoRAHFSGLangLogprobDiff:
    """Test HF+LoRA vs SGLang+LoRA logprob equivalence on Ascend NPU."""

    @pytest.fixture(scope="class")
    def hf_results(self):
        """Compute reference results using HF Transformers + PEFT."""
        print("\n[HF] Loading base model and LoRA adapter...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, LORA_ADAPTER)
        model.eval()
        
        # Compute logprobs for each prompt
        results = {}
        for prompt in PROMPTS:
            print(f"[HF] Processing prompt: {prompt[:50]}...")
            
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            input_ids = inputs["input_ids"]
            
            with torch.no_grad():
                # Get model outputs
                outputs = model(**inputs)
                logits = outputs.logits
                
                # Compute logprobs for next token prediction
                next_token_logits = logits[0, -1, :]
                logprobs = torch.log_softmax(next_token_logits, dim=-1)
                
                # Get top-k tokens and their logprobs
                topk_logprobs, topk_indices = torch.topk(logprobs, TOP_K)
                
                # Convert to dictionary
                token_logprobs = {}
                for idx, logprob in zip(topk_indices.tolist(), topk_logprobs.tolist()):
                    token_str = tokenizer.decode([idx])
                    token_logprobs[token_str] = logprob
            
            results[prompt] = {
                "token_logprobs": token_logprobs,
                "input_ids": input_ids[0].tolist(),
            }
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        
        return results

    @pytest.fixture(scope="class")
    def sglang_server(self):
        """Start SGLang server with LoRA support on Ascend NPU."""
        print("\n[SGLang] Starting server with LoRA support on Ascend NPU...")
        
        # Server configuration
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'test-lora':'{LORA_ADAPTER}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
        }
        
        # Start server
        server_process = run_sglang_server(**server_config)
        
        # Wait for server to be ready
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        # Teardown
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    @pytest.fixture(scope="class")
    def sglang_results(self, sglang_server):
        """Compute results using SGLang with LoRA on Ascend NPU."""
        print("\n[SGLang] Computing logprobs for prompts...")
        
        results = {}
        for prompt in PROMPTS:
            print(f"[SGLang] Processing prompt: {prompt[:50]}...")
            
            # Call SGLang API to get logprobs
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="test-lora",
                max_tokens=1,
                logprobs=TOP_K,
                temperature=0,
            )
            
            # Extract logprobs from response
            token_logprobs = {}
            if "logprobs" in response and "content" in response["logprobs"]:
                for token_info in response["logprobs"]["content"]:
                    if "top_logprobs" in token_info:
                        for item in token_info["top_logprobs"]:
                            token_str = item.get("token", "")
                            logprob = item.get("logprob", 0.0)
                            token_logprobs[token_str] = logprob
            
            results[prompt] = {
                "token_logprobs": token_logprobs,
            }
        
        return results

    def test_logprob_equivalence(self, hf_results, sglang_results):
        """Test that HF and SGLang produce equivalent logprobs."""
        print("\n[Test] Comparing HF and SGLang logprobs...")
        
        max_diff = 0.0
        total_comparisons = 0
        failed_comparisons = 0
        
        for prompt in PROMPTS:
            hf_data = hf_results[prompt]
            sgl_data = sglang_results[prompt]
            
            hf_logprobs = hf_data["token_logprobs"]
            sgl_logprobs = sgl_data["token_logprobs"]
            
            # Compare logprobs for common tokens
            common_tokens = set(hf_logprobs.keys()) & set(sgl_logprobs.keys())
            
            for token in common_tokens:
                hf_lp = hf_logprobs[token]
                sgl_lp = sgl_logprobs[token]
                diff = abs(hf_lp - sgl_lp)
                
                max_diff = max(max_diff, diff)
                total_comparisons += 1
                
                if diff > LOGPROB_DIFF_THRESHOLD:
                    failed_comparisons += 1
                    print(f"  ⚠️  Token '{token}': HF={hf_lp:.4f}, SGL={sgl_lp:.4f}, diff={diff:.4f}")
        
        print(f"\n[Summary]")
        print(f"  Total comparisons: {total_comparisons}")
        print(f"  Failed comparisons: {failed_comparisons}")
        print(f"  Maximum difference: {max_diff:.6f}")
        print(f"  Threshold: {LOGPROB_DIFF_THRESHOLD}")
        
        # Assert that differences are within threshold
        assert max_diff <= LOGPROB_DIFF_THRESHOLD, (
            f"Maximum logprob difference ({max_diff:.6f}) exceeds threshold "
            f"({LOGPROB_DIFF_THRESHOLD})"
        )
        
        assert failed_comparisons == 0, (
            f"{failed_comparisons} token comparisons failed the threshold check"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
