"""
NPU LoRA Generated Tests Package

This package contains generated test cases for NPU (Ascend) LoRA testing based on
the gap analysis between NPU and GPU test coverage.

Generated from: NPU_vs_GPU_LoRA_Test_Gap_Analysis.md
Generation Date: 2026-05-12
"""

__version__ = "1.0.0"

# Test modules
TEST_MODULES = [
    # High Priority Tests (8)
    "test_npu_lora_hf_sgl_logprob_diff",
    "test_npu_lora_eviction",
    "test_npu_lora_update",
    "test_npu_multi_lora_backend",
    "test_npu_lora_tp",
    "test_npu_embedding_lora_support",
    "test_npu_lora_moe_tp_logprob_diff",
    "test_npu_lora_qwen3",
    
    # Medium Priority Tests (Partial)
    "test_npu_lora_radix_cache",
    "test_npu_lora_tied_lm_head",
    "test_npu_lora_qwen3_5_4b_logprob_diff",
    "test_npu_lora_deepseek_v3_base_logprob_diff",
    "test_npu_lora_kimi_k25_logprob_diff",
    "test_npu_lora_gpt_oss_20b_logprob_diff",
]


def get_test_modules():
    """Get list of all test modules."""
    return TEST_MODULES.copy()


def get_high_priority_tests():
    """Get high priority test modules."""
    return TEST_MODULES[:8]


def get_medium_priority_tests():
    """Get medium priority test modules."""
    return TEST_MODULES[8:]
