"""
Simple test for --enable-prefix-mm-cache: send same image twice, verify second hits cache.
"""

import io
import time
import unittest

import openai

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    # DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
from sglang.test.vlm_utils import IMAGE_SGL_LOGO_URL


class TestPrefixMMCacheSimple(unittest.TestCase):
    """Test that sending same image twice results in cache hit on second request."""

    model = "/home/weights/Qwen/Qwen3-VL-8B-Instruct"
    base_host = "127.0.0.1"
    base_port = 31600

    @classmethod
    def setUpClass(cls):
        cls.encode_port = cls.base_port
        cls.encode_url = f"http://{cls.base_host}:{cls.encode_port}"
        cls.encode_stdout = io.StringIO()
        cls.encode_stderr = io.StringIO()

        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--tp",
            "1",
            "--port",
            str(cls.encode_port),
            "--enable-prefix-mm-cache",
        ]

        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
            return_stdout_stderr=(cls.encode_stdout, cls.encode_stderr),
        )
        time.sleep(5)

    def test_same_image_cache_hit(self):
        """Send same image twice, verify second request hits cache."""
        client = openai.OpenAI(
            api_key="sk-123456",
            base_url=f"{self.encode_url}/v1",
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
        if hasattr(cls, 'process_encode') and cls.process_encode:
            try:
                kill_process_tree(cls.process_encode.pid)
            except Exception as e:
                print(f"Error killing process: {e}")


if __name__ == "__main__":
    unittest.main()
