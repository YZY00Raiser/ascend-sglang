import unittest

import requests

from sglang.srt.utils import kill_process_tree
# from sglang.test.ascend.test_ascend_utils import (
#     IMAGES_LOGO_PATH,
#     IMAGES_MAN_PATH,
#     QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH,
# )
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH = "/home/weights/Qwen/Qwen3-VL-8B-Instruct"

IMAGES_LOGO_PATH = "/home/y30082119/man.png"
IMAGES_MAN_PATH = "/home/y30082119/man.png"
# MODEL = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH

# image
IMAGE_MAN_IRONING_URL = IMAGES_MAN_PATH
IMAGE_SGL_LOGO_URL = IMAGES_LOGO_PATH


class TestLimitMMDatePerRequest(TestVLMModels, CustomTestCase):
    """Testcase: Configuring Multi-Modal to send different multimodal inference requests,
       each containing multiple multimodal input data.

    [Test Category] Parameter
    [Test Target] --mm-max-concurrent-calls; --mm-per-request-timeout; --enable-broadcast-mm-inputs-process; --limit-mm-data-per-request
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--encoder-only",
                "--enable-prefix-mm-cache",
                "--tp-size",
                1,
                # "--base-gpu-id",
                # "1"
            ],

        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _run_multi_turn_request(self):
        # Input video and image respectively
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in a sentence.",
                    },
                ],
            },
        ]
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        )
        assert response.status_code == 200
        print(response.json())

    def test_vlm(self):
        self._run_multi_turn_request()


if __name__ == "__main__":
    unittest.main()
