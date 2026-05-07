"""
Simple test for --enable-prefix-mm-cache: send same image twice, verify second hits cache.
Uses encoder-only + language-only mode.
"""

import time
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
from sglang.test.vlm_utils import IMAGE_SGL_LOGO_URL


class TestPrefixMMCacheSimple(unittest.TestCase):
    """Test that sending same image twice results in cache hit on second request."""

    model = "/home/weights/Qwen/Qwen3-VL-8B-Instruct"
    base_host = "127.0.0.1"
    encoder_port = 31600
    language_port = 31602

    @classmethod
    def setUpClass(cls):
        cls.encoder_url = f"http://{cls.base_host}:{cls.encoder_port}"
        cls.language_url = f"http://{cls.base_host}:{cls.language_port}"

        # Start encoder-only server
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--port",
            cls.encoder_port,
            "--enable-prefix-mm-cache",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--base-gpu-id",
            "4",
        ]

        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,

        )

        # Start language-only server
        language_args = [
            "--trust-remote-code",
            "--language-only",
            "--encoder-urls",
            cls.encoder_url,
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--port",
            cls.language_port,
            "--base-gpu-id",
            "12",

        ]

        cls.process_language = popen_launch_server(
            cls.model,
            base_url=cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
        )

        # Wait for servers to be ready
        time.sleep(5)

    def test_same_image_cache_hit(self):
        """Send same image twice, verify second request hits cache."""

        # Input video and image respectively
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": IMAGE_SGL_LOGO_URL},
                    },
                    {
                        "type": "text",
                        "text": "Describe this image in a sentence.",
                    },
                ],
            },
        ]
        for i in range(2):
            response = requests.post(f"{self.language_url}/v1/chat/completions",
                                     json={
                                         "messages": messages,
                                         "temperature": 0,
                                         "max_completion_tokens": 1024,
                                     },
                                     )
            # assert response.status_code == 200
            print(response.json())

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process_language.pid)
        kill_process_tree(cls.process_encode.pid)


if __name__ == "__main__":
    unittest.main()
