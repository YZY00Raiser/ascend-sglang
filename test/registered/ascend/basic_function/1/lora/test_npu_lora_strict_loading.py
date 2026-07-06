import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
# LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH,
# LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH,
# LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
# )
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)
LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH = "/home/weights/Llama-3.2-1B-Instruct"
LLAMA_3_2_1B_INSTRUCT_TOOL_CALLING_LORA_WEIGHTS_PATH = "/home/weights/codelion/Llama-3.2-1B-Instruct-tool-calling-lora"
LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH = "/home/weights/codelion/FastLlama-3.2-LoRA"
Qwen3 = "/home/weights/lora-diff-Qwen3-8B"

import torch
import os

adapter_path = "/home/weights/codelion/FastLlama-3.2-LoRA"
weight_file = os.path.join(adapter_path, "adapter_model.bin")
'''

# 加载权重
state_dict = torch.load(weight_file, map_location="cpu")

# 将 q_proj 改为 nonexistent_proj
new_state_dict = {}
for key, value in state_dict.items():
    if "q_proj" in key:
        new_key = key.replace("q_proj", "nonexistent_proj")
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

# 保存
torch.save(new_state_dict, weight_file)

# 同时修改 adapter_config.json
import json
config_path = os.path.join(adapter_path, "adapter_config.json")
with open(config_path, "r") as f:
    config = json.load(f)

config["target_modules"] = ["nonexistent_proj", "k_proj", "v_proj", "o_proj"]
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)
'''


class TestLora1(CustomTestCase):
    """Testcase：Verify set the --max-load-rank, --lora-backend parameter, load lora that match the number of ranks,
    inference request successful.

    [Test Category] Parameter
    [Test Target] --max-load-rank, --lora-backend
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--max-lora-rank",
            "16",
            "--lora-target-modules",
            "gate_proj",
            "--lora-backend",
            "ascend",
            "--lora-strict-loading",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--base-gpu-id",
            "2",

        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_strict_loading(self):
        response = requests.post(
            DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
            json={"lora_name": "lora_a", "lora_path": self.lora_a, },
        )
        print(response.json())

        # response = requests.post(
        #     f"{DEFAULT_URL_FOR_TEST}/generate",
        #     json={
        #         "text": "The capital of France is",
        #         "sampling_params": {
        #             "temperature": 0,
        #             "max_new_tokens": 32,
        #         },
        #         # "lora_path": "lora_a",
        #     },
        # )
        self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        # response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        # self.assertEqual(response.status_code, 200)


'''
class TestLora2(CustomTestCase):
    """Testcase：Verify set the --max-load-rank, --lora-backend parameter, load lora that match the number of ranks,
    inference request successful.

    [Test Category] Parameter
    [Test Target] --max-load-rank, --lora-backend
    """

    lora_a = LLAMA_3_2_1B_INSTRUCT_TOOL_FAST_LORA_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        other_args = [
            "--enable-lora",
            "--lora-path",
            f"lora_a={cls.lora_a}",
            "--lora-backend",
            "ascend",
            # "--lora-strict-loading",
            # "--no-lora-strict-loading",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]
        cls.process = popen_launch_server(
            LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_lora_max_lora_rank(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
                # "lora_path": "lora_a",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("Paris", response.text)
        response = requests.get(DEFAULT_URL_FOR_TEST + "/server_info")
        self.assertEqual(response.status_code, 200)
'''

if __name__ == "__main__":
    unittest.main()
