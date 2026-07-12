import unittest
from types import SimpleNamespace


from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(est_time=200, suite="full-4-npu-a3", nightly=True)


class TestDtypeAuto(CustomTestCase):

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url="127.0.0.1:23333",
            model="/home/weights/Llama-3.1-8B-Instruct",
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"Eval accuracy of GSM8K: {metrics=}")
        self.assertGreater(metrics["score"], 0.74)

if __name__ == "__main__":
    unittest.main()
