import unittest

import requests

from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    CustomTestCase,
)

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

IMAGES_LOGO_PATH = "/home/y30082119/man.png"
IMAGES_MAN_PATH = "/home/y30082119/man.png"
# MODEL = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH

# image
IMAGE_MAN_IRONING_URL = IMAGES_MAN_PATH
IMAGE_SGL_LOGO_URL = IMAGES_LOGO_PATH


class TestLimitMMDatePerRequest(TestVLMModels, CustomTestCase):
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
        response = requests.post("http://127.0.0.1:30002/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        )
        # assert response.status_code == 200
        print(response.json())

    def _run_multi_turn_request1(self):
        # Enter two images
        messages2 = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_MAN_IRONING_URL},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_SGL_LOGO_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this video in a sentence.",
                    },
                ],
            },
        ]
        response2 = requests.post(
            self.base_url + "/v1/chat/completions",
            json={
                "messages": messages2,
                "temperature": 0,
                "max_completion_tokens": 1024,
            },
        )
        assert response2.status_code == 400

    def test_vlm(self):
        self._run_multi_turn_request()
        # self._run_multi_turn_request1()
        # self._run_parallel_two_requests()


if __name__ == "__main__":
    unittest.main()
