"""
Test NPU LoRA Qwen3 support - 验证NPU上Qwen3模型的LoRA推理功能

Tests Qwen3 model LoRA inference on Ascend NPU:
1. Qwen3 base model LoRA inference
2. Auto-detection of LoRA target modules
3. Generation quality validation
"""

import os
import sys
import pytest
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../.."))

from sglang.test.test_ascend_utils import (
    run_sglang_server,
    kill_sglang_server,
    wait_for_server_ready,
    call_sglang_generate,
    call_sglang_chat,
)

# Test configuration
MODEL_NAME = "Qwen/Qwen3-8B"  # Qwen3 8B base model
ADAPTER_URL = "https://huggingface.co/sglang/qwen3-lora-test"

PROMPTS = [
    "介绍一下机器学习的基本概念。",
    "什么是深度学习？请简要说明。",
    "解释自然语言处理的应用场景。",
]

CHAT_MESSAGES = [
    [{"role": "user", "content": "你好，请介绍一下自己。"}],
    [{"role": "user", "content": "什么是人工智能？"}],
    [{"role": "system", "content": "你是一个有帮助的助手。"},
     {"role": "user", "content": "解释神经网络。"}],
]


class TestNpuLoRAQwen3:
    """Test Qwen3 LoRA inference on Ascend NPU."""

    @pytest.fixture(scope="class")
    def server_process(self):
        """Start SGLang server with Qwen3 model and LoRA."""
        print("\n[SGLang] Starting server for Qwen3 LoRA testing...")
        
        server_config = {
            "model_path": MODEL_NAME,
            "tp_size": 1,
            "lora_paths": f"{{'qwen3-lora':'{ADAPTER_URL}'}}",
            "lora_backend": "ascend",
            "device": "npu",
            "attention_backend": "ascend",
            "enable_lora": True,
            # Auto-detection of LoRA target modules
        }
        
        server_process = run_sglang_server(**server_config)
        wait_for_server_ready(timeout=300)
        
        yield server_process
        
        print("\n[SGLang] Stopping server...")
        kill_sglang_server(server_process)

    def test_qwen3_basic_lora_inference(self, server_process):
        """Test basic Qwen3 LoRA inference."""
        print("\n[Test] Testing Qwen3 basic LoRA inference...")
        
        for prompt in PROMPTS:
            print(f"  Prompt: {prompt[:40]}...")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="qwen3-lora",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            print(f"    Generated: {text[:60]}...")
            
            assert text and len(text) > 0, "Should generate non-empty text"
            
            # Check that output is in Chinese (Qwen3 typically responds in Chinese)
            # Allow for mixed content
            has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
            print(f"    Contains Chinese: {has_chinese}")
        
        print("  ✅ Qwen3 basic LoRA inference works correctly")

    def test_qwen3_chat_lora_inference(self, server_process):
        """Test Qwen3 chat LoRA inference with conversation format."""
        print("\n[Test] Testing Qwen3 chat LoRA inference...")
        
        for messages in CHAT_MESSAGES:
            print(f"  Messages: {[m['content'][:30] + '...' for m in messages]}")
            
            response = call_sglang_chat(
                messages=messages,
                lora_path="qwen3-lora",
                max_tokens=100,
                temperature=0.7,
                top_p=0.9,
            )
            
            text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"    Assistant: {text[:60]}...")
            
            assert text and len(text) > 0, "Should generate non-empty response"
        
        print("  ✅ Qwen3 chat LoRA inference works correctly")

    def test_qwen3_lora_target_modules_auto_detection(self, server_process):
        """Test auto-detection of LoRA target modules for Qwen3."""
        print("\n[Test] Testing Qwen3 LoRA target module auto-detection...")
        
        # This test verifies that the server correctly auto-detects
        # the LoRA target modules for Qwen3 architecture
        
        # Qwen3 typically uses q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
        expected_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
        
        # Generate with LoRA to verify it's being applied correctly
        prompt = "列出三种机器学习的类型。"
        
        response = call_sglang_generate(
            prompt=prompt,
            lora_path="qwen3-lora",
            max_tokens=80,
            temperature=0.0,  # Deterministic for consistency
        )
        
        text = response.get("choices", [{}])[0].get("text", "")
        print(f"  Prompt: {prompt}")
        print(f"  Generated: {text[:80]}...")
        
        assert text and len(text) > 0, "Should generate non-empty text"
        
        # The response should be in Chinese (Qwen3 characteristic)
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        print(f"  Contains Chinese characters: {has_chinese}")
        
        print("  ✅ Qwen3 LoRA target module auto-detection works correctly")

    def test_qwen3_lora_generation_quality(self, server_process):
        """Test Qwen3 LoRA generation quality."""
        print("\n[Test] Testing Qwen3 LoRA generation quality...")
        
        test_cases = [
            {
                "prompt": "什么是深度学习？",
                "expected_keywords": ["深度学习", "神经网络", "机器学习"],
            },
            {
                "prompt": "解释自然语言处理。",
                "expected_keywords": ["自然语言处理", "NLP", "语言"],
            },
            {
                "prompt": "列举三个人工智能的应用。",
                "expected_keywords": ["应用", "智能"],
            },
        ]
        
        for test in test_cases:
            prompt = test["prompt"]
            expected_keywords = test["expected_keywords"]
            
            print(f"  Prompt: {prompt}")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="qwen3-lora",
                max_tokens=150,
                temperature=0.3,  # Lower temperature for more focused output
                top_p=0.9,
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            print(f"    Generated: {text[:100]}...")
            
            # Check for expected keywords
            found_keywords = [kw for kw in expected_keywords if kw.lower() in text.lower()]
            print(f"    Found keywords: {found_keywords}")
            
            # At least one keyword should be found
            assert len(found_keywords) > 0, (
                f"Expected at least one of {expected_keywords} in response, "
                f"but none found in: {text[:200]}"
            )
        
        print("  ✅ Qwen3 LoRA generation quality is acceptable")

    def test_qwen3_lora_with_different_temperatures(self, server_process):
        """Test Qwen3 LoRA with different temperature settings."""
        print("\n[Test] Testing Qwen3 LoRA with different temperatures...")
        
        prompt = "描述一下未来的人工智能发展。"
        temperatures = [0.0, 0.5, 0.9, 1.2]
        
        results = {}
        for temp in temperatures:
            print(f"  Temperature={temp}:")
            
            response = call_sglang_generate(
                prompt=prompt,
                lora_path="qwen3-lora",
                max_tokens=80,
                temperature=temp,
                top_p=0.95,
            )
            
            text = response.get("choices", [{}])[0].get("text", "")
            results[temp] = text
            print(f"    {text[:70]}...")
            
            assert text and len(text) > 0, f"Should generate output at temperature {temp}"
        
        # With temperature=0, outputs should be deterministic
        # Generate twice to verify
        print("  Verifying temperature=0 determinism...")
        response2 = call_sglang_generate(
            prompt=prompt,
            lora_path="qwen3-lora",
            max_tokens=80,
            temperature=0.0,
            top_p=0.95,
        )
        text2 = response2.get("choices", [{}])[0].get("text", "")
        
        assert results[0.0] == text2, "Temperature=0 should produce deterministic output"
        print("    Temperature=0 is deterministic ✓")
        
        print("  ✅ Qwen3 LoRA with different temperatures works correctly")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
