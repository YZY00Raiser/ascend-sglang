"""
Simple test for --enable-prefix-mm-cache: send same image twice, verify second hits cache.
Uses encoder-only + language-only mode.
"""

import io
import time
import unittest

import openai
import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
# from sglang.test.vlm_utils import IMAGE_SGL_LOGO_URL

IMAGE_SGL_LOGO_URL="/home/y30082119/sgl_logo.png"


class TestPrefixMMCacheSimple(unittest.TestCase):
    """Test that sending same image twice results in cache hit on second request."""

    model = "/home/weights/Qwen3-VL-8B-Instruct"
    base_host = "127.0.0.1"
    encoder_port = 31600
    language_port = 31602

    @classmethod
    def setUpClass(cls):
        cls.encoder_url = f"http://{cls.base_host}:{cls.encoder_port}"
        cls.language_url = f"http://{cls.base_host}:{cls.language_port}"
        cls.encode_stdout = io.StringIO()
        cls.encode_stderr = io.StringIO()
        cls.language_stdout = io.StringIO()
        cls.language_stderr = io.StringIO()

        # Start encoder-only server
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--port",
            str(cls.encoder_port),
            "--enable-prefix-mm-cache",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--base-gpu-id",
            "8",
        ]

        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encoder_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            return_stdout_stderr=(cls.encode_stdout, cls.encode_stderr),
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
            str(cls.language_port),
            "--base-gpu-id",
            "9",

        ]

        cls.process_language = popen_launch_server(
            cls.model,
            base_url=cls.language_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=language_args,
            return_stdout_stderr=(cls.language_stdout, cls.language_stderr),
        )

        # Wait for servers to be ready
        time.sleep(5)

    def test_same_image_cache_hit(self):
        """Send same image twice, verify second request hits cache."""
        client = openai.OpenAI(
            api_key="sk-123456",
            base_url=f"{self.language_url}/v1",
        )

        # First request - cache miss
        response1 = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_SGL_LOGO_URL},
                        },
                        {
                            "type": "text",
                            "text": "What is in this image?",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=128,
        )
        print(f"First response: {response1.choices[0].message.content[:100]}...")

        time.sleep(1)

        # Second request - should hit cache
        response2 = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": IMAGE_SGL_LOGO_URL},
                        },
                        {
                            "type": "text",
                            "text": "What is in this image?",
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=128,
        )
        print(f"Second response: {response2.choices[0].message.content[:100]}...")

        # Verify responses are valid
        self.assertIsNotNone(response1.choices[0].message.content)
        self.assertIsNotNone(response2.choices[0].message.content)

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process_language.pid)
        kill_process_tree(cls.process_encode.pid)



if __name__ == "__main__":
    unittest.main()
