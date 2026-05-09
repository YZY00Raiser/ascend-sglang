# Copyright 2023-2024 SGLang Team
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
NPU LoRA Logprob Difference Test (SGLang vs HuggingFace)

This test compares log probabilities between SGLang+LoRA on NPU and HuggingFace+LoRA.
Test cases are ported from GPU test_lora_hf_sgl_logprob_diff.py for NPU compatibility.

[Test Category] Precision
[Test Target] Logprob consistency between NPU SGLang and HF baseline

Note: This test requires either:
  1. Dual environment (NPU for SGLang + GPU for HF baseline)
  2. Or run SGLang only mode for regression testing
"""

import multiprocessing as mp
import os
import unittest
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-2-npu-a3",
    nightly=True,
)

# Test configuration constants
BASE_MODEL = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
LORA_PATH = LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH
DISABLE_CUDA_GRAPH = True
LORA_TARGET_MODULES = ["all"]
LOGPROB_THRESHOLD = 1e-01
MAX_NEW_TOKENS = 32

# Default test prompts
DEFAULT_TEST_PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Write a short story.",
    "What are the main components of a computer?",
]

# Formatting constants
DIVIDER_WIDTH = 80
SECTION_CHAR = "="
SUBSECTION_CHAR = "-"


def print_section_header(title: str):
    """Print a major section header."""
    print("\n" + SECTION_CHAR * DIVIDER_WIDTH)
    print(title)
    print(SECTION_CHAR * DIVIDER_WIDTH)


def print_subsection_header(title: str):
    """Print a subsection header."""
    print(f"\n{SUBSECTION_CHAR * 40}")
    print(f"{title}")
    print(SUBSECTION_CHAR * 40)


def print_config_info(title: str, config: Dict[str, Any]):
    """Print configuration information in a consistent format."""
    print_section_header(title)
    for key, value in config.items():
        print(f"  {key}: {value}")


def compare_logprobs_for_type(
    sglang_logprobs: torch.Tensor, hf_logprobs: torch.Tensor, logprob_type: str
) -> Dict[str, Any]:
    """
    Compare logprobs for a specific type (prefill or decode).

    Args:
        sglang_logprobs: SGLang log probabilities
        hf_logprobs: HuggingFace log probabilities
        logprob_type: Type of logprobs ("prefill" or "decode")

    Returns:
        Dictionary containing comparison statistics
    """
    diff = torch.abs(sglang_logprobs - hf_logprobs)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    shape = list(sglang_logprobs.shape)
    matches_threshold = max_diff < LOGPROB_THRESHOLD

    return {
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "shape": shape,
        "matches_threshold": matches_threshold,
        "type": logprob_type,
    }


def print_logprob_comparison(comparison: Dict[str, Any]):
    """Print logprob comparison results in a consistent format."""
    logprob_type = comparison["type"].capitalize()
    print(f"\n{logprob_type} logprobs:")
    print(f"  Shape:           {comparison['shape']}")
    print(f"  Max difference:  {comparison['max_diff']:.6e}")
    print(f"  Mean difference: {comparison['mean_diff']:.6e}")

    status = "PASS" if comparison["matches_threshold"] else "FAIL"
    print(f"  Status:          {status} (threshold: {LOGPROB_THRESHOLD:.0e})")


def prepare_lora_paths_per_prompt(
    lora_paths: List[str], num_prompts: int
) -> List[Optional[str]]:
    """
    Prepare LoRA paths for each prompt by cycling through available LoRAs.

    Args:
        lora_paths: List of available LoRA adapter paths
        num_prompts: Number of prompts to generate LoRA paths for

    Returns:
        List of LoRA paths (one per prompt), or None values if no LoRAs
    """
    if not lora_paths:
        return [None] * num_prompts

    return [lora_paths[i % len(lora_paths)] for i in range(num_prompts)]


def run_sglang_with_lora(
    model_path: str,
    lora_paths: List[str],
    prompts: List[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
    port: int,
) -> Dict[str, Any]:
    """Run SGLang with LoRA on NPU and return log probabilities."""
    config = {
        "Model": model_path,
        "LoRA paths": lora_paths,
        "Port": port,
        "Number of prompts": len(prompts),
    }
    print_config_info("Running SGLang with LoRA on NPU", config)

    lora_paths_per_prompt = prepare_lora_paths_per_prompt(lora_paths, len(prompts))

    with SRTRunner(
        model_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        tp_size=1,
        lora_paths=lora_paths,
        max_loras_per_batch=len(lora_paths) if lora_paths else 1,
        disable_cuda_graph=DISABLE_CUDA_GRAPH,
        disable_radix_cache=True,
        port=port,
        mem_fraction_static=0.6,
        lora_target_modules=LORA_TARGET_MODULES,
        attention_backend="ascend",
    ) as srt_runner:
        srt_outputs = srt_runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=lora_paths_per_prompt,
        )

    return {
        "top_input_logprobs": srt_outputs.top_input_logprobs,
        "top_output_logprobs": srt_outputs.top_output_logprobs,
        "output_strs": srt_outputs.output_strs,
        "lora_paths": lora_paths_per_prompt,
    }


def run_hf_with_lora(
    model_path: str,
    lora_paths: List[str],
    prompts: List[str],
    max_new_tokens: int,
    torch_dtype: torch.dtype,
) -> Dict[str, Any]:
    """Run HuggingFace with LoRA and return log probabilities."""
    config = {
        "Model": model_path,
        "LoRA paths": lora_paths,
        "Number of prompts": len(prompts),
    }
    print_config_info("Running HuggingFace with LoRA", config)

    lora_paths_per_prompt = prepare_lora_paths_per_prompt(lora_paths, len(prompts))

    with HFRunner(
        model_path,
        torch_dtype=torch_dtype,
        model_type="generation",
        patch_model_do_sample_false=True,
    ) as hf_runner:
        hf_outputs = hf_runner.forward(
            prompts,
            max_new_tokens=max_new_tokens,
            lora_paths=lora_paths_per_prompt,
        )

    return {
        "top_input_logprobs": hf_outputs.top_input_logprobs,
        "top_output_logprobs": hf_outputs.top_output_logprobs,
        "output_strs": hf_outputs.output_strs,
        "lora_paths": lora_paths_per_prompt,
    }


def compare_single_prompt(
    prompt_idx: int,
    sglang_data: Dict[str, Any],
    hf_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare logprobs and outputs for a single prompt.

    Args:
        prompt_idx: Index of the prompt being compared
        sglang_data: SGLang results data
        hf_data: HuggingFace results data

    Returns:
        Dictionary containing all comparison results
    """
    print_subsection_header(f"Prompt {prompt_idx + 1}")
    print(f"LoRA adapter: {sglang_data['lora_paths'][prompt_idx]}")

    result = {
        "prompt_idx": prompt_idx,
        "lora_path": sglang_data["lora_paths"][prompt_idx],
    }

    # Compare prefill (input) logprobs
    sglang_prefill = torch.tensor(sglang_data["top_input_logprobs"][prompt_idx])
    hf_prefill = torch.tensor(hf_data["top_input_logprobs"][prompt_idx])
    prefill_comparison = compare_logprobs_for_type(
        sglang_prefill, hf_prefill, "prefill"
    )
    print_logprob_comparison(prefill_comparison)

    # Store prefill results
    result["prefill_max_diff"] = prefill_comparison["max_diff"]
    result["prefill_mean_diff"] = prefill_comparison["mean_diff"]
    result["prefill_shape"] = prefill_comparison["shape"]
    result["prefill_logprob_match"] = prefill_comparison["matches_threshold"]

    # Compare decode (output) logprobs
    sglang_decode = torch.tensor(sglang_data["top_output_logprobs"][prompt_idx])
    hf_decode = torch.tensor(hf_data["top_output_logprobs"][prompt_idx])
    decode_comparison = compare_logprobs_for_type(sglang_decode, hf_decode, "decode")
    print_logprob_comparison(decode_comparison)

    # Store decode results
    result["decode_max_diff"] = decode_comparison["max_diff"]
    result["decode_mean_diff"] = decode_comparison["mean_diff"]
    result["decode_shape"] = decode_comparison["shape"]
    result["decode_logprob_match"] = decode_comparison["matches_threshold"]

    # Overall logprob match
    result["overall_logprob_match"] = (
        prefill_comparison["matches_threshold"]
        and decode_comparison["matches_threshold"]
    )

    return result


class TestNpuLoRAHFSGLLogprobDifference(CustomTestCase):
    """Test logprob difference between SGLang on NPU and HuggingFace."""

    def test_npu_lora_hf_sgl_logprob_diff(self):
        """
        Compare logprobs between SGLang on NPU and HuggingFace baseline.

        This test can run in two modes:
        1. Full mode: Compare NPU SGLang with HF baseline (requires GPU for HF)
        2. SGLang-only mode: Just verify SGLang runs correctly on NPU
        """
        prompts = DEFAULT_TEST_PROMPTS
        torch_dtype = torch.float16
        lora_paths = [LORA_PATH]

        # Run SGLang on NPU
        sglang_results = run_sglang_with_lora(
            model_path=BASE_MODEL,
            lora_paths=lora_paths,
            prompts=prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            torch_dtype=torch_dtype,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        )

        # Check if we should run HF comparison
        run_hf_comparison = os.environ.get("NPU_LORA_RUN_HF_COMPARISON", "false").lower() == "true"

        if run_hf_comparison:
            # Run HuggingFace for baseline comparison
            hf_results = run_hf_with_lora(
                model_path=BASE_MODEL,
                lora_paths=lora_paths,
                prompts=prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                torch_dtype=torch_dtype,
            )

            # Compare results
            print_section_header("Comparing Results")
            comparison_results = []
            for i in range(len(prompts)):
                result = compare_single_prompt(i, sglang_results, hf_results)
                comparison_results.append(result)

            # Print overall statistics
            self._print_overall_statistics(comparison_results)

            # Assert all logprobs match threshold
            all_match = all(r["overall_logprob_match"] for r in comparison_results)
            self.assertTrue(
                all_match,
                f"Not all logprobs matched threshold {LOGPROB_THRESHOLD:.0e}",
            )
        else:
            print_section_header("SGLang-only Mode (No HF Comparison)")
            print("Set NPU_LORA_RUN_HF_COMPARISON=true to enable HF baseline comparison")

            # Basic validation that SGLang produced outputs
            self.assertEqual(len(sglang_results["output_strs"]), len(prompts))
            for i, output in enumerate(sglang_results["output_strs"]):
                self.assertTrue(len(output) > 0, f"Empty output for prompt {i}")
                print(f"Prompt {i+1} output: {output[:100]}...")

    def _print_overall_statistics(self, results: List[Dict[str, Any]]):
        """Print overall statistics across all prompts."""
        print_section_header("Overall Statistics")

        # Gather statistics
        prefill_max_diffs = [r["prefill_max_diff"] for r in results]
        prefill_mean_diffs = [r["prefill_mean_diff"] for r in results]
        decode_max_diffs = [r["decode_max_diff"] for r in results]
        decode_mean_diffs = [r["decode_mean_diff"] for r in results]

        # Print logprob statistics
        print("\nLogprob Differences:")
        print(f"  Prefill:")
        print(f"    Max of max:   {max(prefill_max_diffs):.6e}")
        print(f"    Mean of max:  {np.mean(prefill_max_diffs):.6e}")
        print(f"    Mean of mean: {np.mean(prefill_mean_diffs):.6e}")

        print(f"  Decode:")
        print(f"    Max of max:   {max(decode_max_diffs):.6e}")
        print(f"    Mean of max:  {np.mean(decode_max_diffs):.6e}")
        print(f"    Mean of mean: {np.mean(decode_mean_diffs):.6e}")

        # Print match statistics
        num_prompts = len(results)
        logprob_match_count = sum(r["overall_logprob_match"] for r in results)
        prefill_match_count = sum(r["prefill_logprob_match"] for r in results)
        decode_match_count = sum(r["decode_logprob_match"] for r in results)

        print(f"\nLogprob Statistics (threshold: {LOGPROB_THRESHOLD:.0e}):")
        overall_status = "PASSED" if logprob_match_count == num_prompts else "FAILED"
        print(f"  Overall logprob: {logprob_match_count}/{num_prompts} {overall_status}")
        print(f"  Prefill logprob: {prefill_match_count}/{num_prompts}")
        print(f"  Decode logprob:  {decode_match_count}/{num_prompts}")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
