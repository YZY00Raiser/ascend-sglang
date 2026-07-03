"""
End-to-end test for --enable-prefix-mm-cache parameter.
Tests the prefix multimodal cache functionality in encoder-only mode.
"""

import os
import re
import threading
import time
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server,
)
from sglang.srt.utils import kill_process_tree

register_cuda_ci(est_time=180, stage="base-c")


class TestPrefixMMCacheE2E(unittest.TestCase):
    """End-to-end test for --enable-prefix-mm-cache with image encoding."""

    model = DEFAULT_SMALL_VLM_MODEL_NAME_FOR_TEST
    base_host = "127.0.0.1"
    encode_port = "30000"
    encode_url = f"http://{base_host}:{encode_port}"
    image_url = "https://raw.githubusercontent.com/sgl-project/sglang/main/test/images/cat.jpg"

    @classmethod
    def setUpClass(cls):
        """Start encoder server with --enable-prefix-mm-cache."""
        encode_args = [
            "--trust-remote-code",
            "--encoder-only",
            "--encoder-transfer-backend",
            "zmq_to_scheduler",
            "--tp",
            "1",
            "--port",
            cls.encode_port,
            "--enable-prefix-mm-cache",
        ]
        cls.process_encode = popen_launch_server(
            cls.model,
            base_url=cls.encode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=encode_args,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up processes."""
        if cls.process_encode:
            try:
                kill_process_tree(cls.process_encode.pid)
            except Exception as e:
                print(f"Error killing process: {e}")

    def _parse_cache_log(self):
        """Parse encode server logs and return cache hit/miss information."""
        # This is a placeholder - actual implementation would parse server logs
        return []

    def test_image_encoding_with_cache(self):
        """Test that image encoding works with prefix mm cache enabled."""
        import requests

        # First request
        response1 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "image",
                "data": [self.image_url],
            },
            timeout=60,
        )
        self.assertEqual(response1.status_code, 200)

        # Second request (should use cache)
        response2 = requests.post(
            f"{self.encode_url}/encode",
            json={
                "modality": "image",
                "data": [self.image_url],
            },
            timeout=60,
        )
        self.assertEqual(response2.status_code, 200)

    def test_health_check(self):
        """Test that encoder server is healthy."""
        import requests

        response = requests.get(f"{self.encode_url}/health", timeout=10)
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
