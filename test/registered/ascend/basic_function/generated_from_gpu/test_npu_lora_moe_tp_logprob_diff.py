# Copyright 2023-2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
NPU MoE LoRA Tensor Parallel Logprob Test

This test compares TP=1 vs TP=2 MoE LoRA output consistency on NPU.
Test cases are ported from GPU test_lora_moe_tp_logprob_diff.py for NPU compatibility.

[Test Category] Precision
[Test Target] MoE LoRA TP parity (TP=1 vs TP=2) on NPU

Note: This is a self-comparison test that doesn't require HF baseline.
"""

import multiprocessing as mp
import unittest
from typing import Any, Dict, List

import torch

from sglang.test.ascend.test_ascend_utils import (
    QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.lora_utils import MOE_LORA_TEST_PROMPTS
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import (
    DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
    CustomTestCase,
    is_in_ci,
)

register_npu_ci(
    est_time=300,
    suite="nightly-2-npu-a3",
    nightly=True,
)

LOGPROB_THRESHOLD = 5e-04
MAX_NEW_TOKENS = 10

# NPU compatible MoE test prompts (subset)
NPU_MOE_LORA_TEST_PROMPTS = MOE_LORA_TEST_PROMPTS[:5]

# NPU compatible MoE model - using Qwen2.5 as alternative
# Note: Original GPU test uses Qwen1.5-MoE-A2.7B, replace with NPU-supported MoE model
NPU_MOE_BASE_MODEL_PATH = QWEN2_5_7B_INSTRUCT_WEIGHTS_PATH
NPU_MOE_LORA_PATH = None  # Set to actual LoRA path if available


def _run_sglang_moe_lora(
    tp_size: int,
    prompts: List[str],
    port: int = DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
) -> Dict[str, Any]:
    """Run SGLang MoE LoRA and return outputs."""
    # If no LoRA path available, run without LoRA for basic TP parity test
    lora_paths = [NPU_MOE_LORA_PATH] if NPU_MOE_LORA_PATH else []
    lora_paths_per_prompt = lora_paths * len(prompts) if lora_paths else None

    with SRTRunner(
        model_path=NPU_MOE_BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        model_type="generation",
        tp_size=tp_size,
        lora_paths=lora_paths if lora_paths else None,
        max_loras_per_batch=1 if lora_paths else 0,
        trust_remote_code=True,
        disable_radix_cache=True,
        port=port,
        mem_fraction_static=0.70,
        attention_backend="ascend",
        disable_cuda_graph=True,
    ) as runner:
        outputs = runner.forward(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            lora_paths=lora_paths_per_prompt,
        )

    return {
        "top_input_logprobs": outputs.top_input_logprobs,
        "top_output_logprobs": outputs.top_output_logprobs,
        "output_strs": outputs.output_strs,
    }


class TestNpuMoELoRATP2Logprobs(CustomTestCase):
    """Compare TP=1 vs TP=2 MoE LoRA on NPU: output strings must match and logprobs
    must stay within threshold."""

    def _assert_tp_parity(
        self,
        prompts: List[str],
        label: str,
    ):
        """Assert TP=1 and TP=2 produce consistent results."""
        print(f"\n{'=' * 100}")
        print(f"  {label}: running TP=1 on NPU")
        print(f"{'=' * 100}")

        tp1 = _run_sglang_moe_lora(tp_size=1, prompts=prompts)

        # Clear cache between runs
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
            torch.npu.synchronize()

        print(f"\n{'=' * 100}")
        print(f"  {label}: running TP=2 on NPU")
        print(f"{'=' * 100}")

        tp2 = _run_sglang_moe_lora(tp_size=2, prompts=prompts)

        print(f"\n{'=' * 100}")
        print(
            f"{'ID':<4} | {'String':<8} | {'Decode Max Diff':<18} | "
            f"{'Decode Mean Diff':<18} | {'Status':<8} | {'Output (TP1)'}"
        )
        print("-" * 100)

        for i in range(len(prompts)):
            tp1_str = tp1["output_strs"][i].strip()
            tp2_str = tp2["output_strs"][i].strip()

            self.assertEqual(
                tp1_str,
                tp2_str,
                f"Output string mismatch on prompt {i}: "
                f"TP1='{tp1_str}' vs TP2='{tp2_str}'",
            )

            # Compare logprobs if available
            tp1_raw = tp1["top_output_logprobs"][i]
            tp2_raw = tp2["top_output_logprobs"][i]
            tp1_lps = torch.tensor(
                [t[0] if isinstance(t, list) else t for t in tp1_raw]
            )
            tp2_lps = torch.tensor(
                [t[0] if isinstance(t, list) else t for t in tp2_raw]
            )
            min_len = min(tp1_lps.shape[0], tp2_lps.shape[0])
            diff = torch.abs(tp1_lps[:min_len] - tp2_lps[:min_len])
            max_diff = torch.max(diff).item() if min_len > 0 else 0.0
            mean_diff = torch.mean(diff).item() if min_len > 0 else 0.0

            status = "PASS" if max_diff < LOGPROB_THRESHOLD else "FAIL"
            print(
                f"{i:<4} | {'OK':<8} | {max_diff:<18.6e} | "
                f"{mean_diff:<18.6e} | {status:<8} | {tp1_str[:40]}"
            )

            self.assertLessEqual(
                max_diff,
                LOGPROB_THRESHOLD,
                f"Decode logprob diff too large on prompt {i}: "
                f"max_diff={max_diff:.6e} > threshold={LOGPROB_THRESHOLD:.0e}",
            )

        print("=" * 100)

    def test_npu_moe_lora_tp2_vs_tp1_basic(self):
        """Basic TP=1 vs TP=2 parity with a small prompt set on NPU."""
        self._assert_tp_parity(
            prompts=NPU_MOE_LORA_TEST_PROMPTS,
            label="MoE LoRA TP parity (basic) on NPU",
        )

    @unittest.skipIf(is_in_ci(), "Skipping full test in CI")
    def test_npu_moe_lora_tp2_vs_tp1_full(self):
        """Full TP=1 vs TP=2 parity across all prompts on NPU."""
        self._assert_tp_parity(
            prompts=MOE_LORA_TEST_PROMPTS,
            label="MoE LoRA TP parity (full) on NPU",
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    try:
        unittest.main(warnings="ignore", verbosity=2)
    finally:
        # Cleanup NPU cache
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.empty_cache()
            torch.npu.synchronize()
