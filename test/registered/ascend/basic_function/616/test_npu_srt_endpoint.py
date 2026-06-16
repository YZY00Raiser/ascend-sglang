"""NPU SRT endpoint tests for logprobs, logit bias, cache tokens, and tokenize/detokenize.

Adapted from test_srt_endpoint.py. Covers API endpoint features not already
covered by existing NPU tests (test_npu_api.py, test_npu_enable_custom_logit_processor.py)."""

import json
import unittest

import requests
from transformers import AutoTokenizer

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="full-1-npu-a3", nightly=True)


class TestNpuSrtEndpoint(CustomTestCase):
    """Test API endpoint features on NPU.

    [Test Category] Core
    [Test Target] /generate, /flush_cache, /tokenize, /detokenize
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_logprob(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                "return_logprob": True,
                "top_logprobs_num": 5,
                "return_text_in_logprobs": True,
                "logprob_start_len": 0,
            },
        )
        self.assertEqual(response.status_code, 200)
        meta = response.json()["meta_info"]
        self.assertIn("input_token_logprobs", meta)
        self.assertIn("output_token_logprobs", meta)

    def test_cache_tokens(self):
        for _ in range(2):
            response = requests.post(self.base_url + "/flush_cache")
            self.assertEqual(response.status_code, 200)

        def check_cached(input_ids):
            response = requests.post(
                self.base_url + "/generate",
                json={
                    "input_ids": list(input_ids),
                    "sampling_params": {"max_new_tokens": 1},
                },
            )
            return response.json()["meta_info"]["cached_tokens"]

        self.assertEqual(check_cached(range(0, 100)), 0)
        self.assertEqual(check_cached(range(0, 10000)), 100)
        self.assertEqual(check_cached(range(0, 10000)), 9999)

    def test_get_server_info(self):
        response = requests.get(self.base_url + "/server_info")
        self.assertEqual(response.status_code, 200)
        info = response.json()
        self.assertIn("max_total_num_tokens", info)
        self.assertIn("version", info)

    def test_logit_bias(self):
        target_token_id = 60704  # Paris for Llama-3.2-1B-Instruct
        logit_bias = {str(target_token_id): 100.0}
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": 4,
                    "logit_bias": logit_bias,
                },
                "return_logprob": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        sampled_tokens = [x[1] for x in response.json()["meta_info"]["output_token_logprobs"]]
        self.assertTrue(
            all(x == target_token_id for x in sampled_tokens),
            f"Expected all tokens to be {target_token_id}, got {sampled_tokens}",
        )

    def test_forbidden_token(self):
        forbidden_token_id = 23994  # rice for Llama-3.2-1B-Instruct
        logit_bias = {str(forbidden_token_id): -100.0}
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Only output 'rice' in lowercase: rice",
                "sampling_params": {
                    "temperature": 1.0,
                    "max_new_tokens": 50,
                    "logit_bias": logit_bias,
                },
                "return_logprob": True,
            },
        )
        self.assertEqual(response.status_code, 200)
        sampled_tokens = [x[1] for x in response.json()["meta_info"]["output_token_logprobs"]]
        self.assertNotIn(forbidden_token_id, sampled_tokens)


class TestNpuTokenizeDetokenize(CustomTestCase):
    """Test tokenize/detokenize endpoints on NPU.

    [Test Category] Core
    [Test Target] /tokenize, /detokenize
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.tokenize_url = f"{cls.base_url}/tokenize"
        cls.detokenize_url = f"{cls.base_url}/detokenize"
        cls.session = requests.Session()
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=(
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
            ),
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.session.close()

    def test_tokenize_various_inputs(self):
        single = "Hello SGLang world! 123, 中文."
        multi = ["First sentence.", "Second sentence."]
        scenarios = [
            {"prompt": single, "add_special_tokens": True},
            {"prompt": single, "add_special_tokens": False},
            {"prompt": multi, "add_special_tokens": True},
            {"prompt": multi, "add_special_tokens": False},
        ]
        for case in scenarios:
            payload = {"model": self.model, "prompt": case["prompt"]}
            if "add_special_tokens" in case:
                payload["add_special_tokens"] = case["add_special_tokens"]
            resp = self.session.post(self.tokenize_url, json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIsInstance(data["tokens"], list)
            self.assertIsInstance(data["count"], (int, list))

    def test_detokenize_roundtrip(self):
        text = "Verify detokenization round trip."
        t0 = self.session.post(
            self.tokenize_url,
            json={"model": self.model, "prompt": text, "add_special_tokens": False},
        ).json()["tokens"]
        resp = self.session.post(
            self.detokenize_url,
            json={"model": self.model, "tokens": t0, "skip_special_tokens": True},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["text"], text)

    def test_tokenize_invalid_type(self):
        resp = self.session.post(
            self.tokenize_url, json={"model": self.model, "prompt": 12345}
        )
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
