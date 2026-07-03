import multiprocessing as mp
import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.lora_utils import (
    CI_MULTI_LORA_MODELS,
    run_lora_batch_splitting_equivalence_test,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=100, stage="extra-a", runner_config="1-gpu-small")
register_amd_ci(est_time=100, suite="stage-b-test-1-gpu-small-amd")


class TestLoRADrainWaitThreshold(CustomTestCase):
    def test_drain_disabled(self):
        run_lora_batch_splitting_equivalence_test(
            model_cases=CI_MULTI_LORA_MODELS,
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            lora_drain_wait_threshold=0.0,
        )

    def test_drain_enabled_low_threshold(self):
        run_lora_batch_splitting_equivalence_test(
            model_cases=CI_MULTI_LORA_MODELS,
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            lora_drain_wait_threshold=0.1,
        )

    def test_drain_enabled_medium_threshold(self):
        run_lora_batch_splitting_equivalence_test(
            model_cases=CI_MULTI_LORA_MODELS,
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            lora_drain_wait_threshold=3.0,
        )

    def test_drain_enabled_high_threshold(self):
        run_lora_batch_splitting_equivalence_test(
            model_cases=CI_MULTI_LORA_MODELS,
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            lora_drain_wait_threshold=100.0,
        )

    def test_drain_with_deterministic_inference(self):
        run_lora_batch_splitting_equivalence_test(
            model_cases=CI_MULTI_LORA_MODELS,
            attention_backend="torch_native",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            lora_drain_wait_threshold=3.0,
            enable_deterministic_inference=True,
        )


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
