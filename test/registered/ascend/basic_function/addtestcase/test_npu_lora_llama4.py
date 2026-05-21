from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestLlama4(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the meta-llama/Llama-4-Scout-17B-16E-Instruct model on the GSM8K dataset is no less than 0.9.

    [Test Category] Model
    [Test Target] meta-llama/Llama-4-Scout-17B-16E-Instruct
    """

    model = LLAMA_4_SCOUT_17B_16E_INSTRUCT_WEIGHTS_PATH
    accuracy = 0.9
    timeout_for_server_launch = 1000
    other_args = [
        "--chat-template",
        "llama-4",
        "--tp-size",
        4,
        "--mem-fraction-static",
        "0.9",
        "--context-length",
        "8192",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]


class TestLlama4LoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        process = popen_launch_server(
            model.model,
            self.base_url,
            timeout=3 * DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--mem-fraction-static",
                0.8,
                # "--cuda-graph-max-bs",
                # 16,
                "--disable-cuda-graph",
                "--disable-radix-cache",
                "--enable-lora",
                "--max-lora-rank",
                "64",
                "--lora-target-modules",
                "qkv_proj",
                "o_proj",
                "gate_up_proj",
                "down_proj",
                "--tp-size",
                str(model.tp_size),
                "--context-length",
                "262144",
                "--attention-backend",
                "ascend",
            ],
        )

        # json_schema = json.dumps({
        #     "type": "object",
        #     "properties": {
        #         "name": {"type": "string"},
        #         "age": {"type": "integer"},
        #         "city": {"type": "string"},
        #     },
        #     "required": ["name", "age", "city"],
        #
        # })
        # response = requests.post(
        #     f"{DEFAULT_URL_FOR_TEST}/generate",
        #     json={
        #         "text": "Generate person information",
        #         "sampling_params": {
        #             "temperature": 0.3,
        #             "max_new_tokens": 128,
        #             "json_schema": json_schema,
        #         },
        #         "lora_path": "lora_a",
        #     },
        # )
        # self.assertEqual(response.status_code, 200)
        # result = response.json()
        # parsed_json = json.loads(result["text"])
        # self.assertIn("name", parsed_json)
        # self.assertIn("age", parsed_json)
        # self.assertIn("city", parsed_json)

    except Exception as e:
    print(f"Error testing {model.model}: {e}")
    self.fail(f"Test failed for {model.model}: {e}")

finally:
# Ensure process cleanup happens regardless of success/failure
if process is not None and process.poll() is None:
    print(f"Cleaning up process {process.pid}")
    try:
        kill_process_tree(process.pid)
    except Exception as e:
        print(f"Error killing process: {e}")


def test_bringup(self):


if __name__ == "__main__":
    unittest.main()
