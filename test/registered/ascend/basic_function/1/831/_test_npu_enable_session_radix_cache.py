"""End-to-end behavior test for --enable-session-radix-cache (UnifiedRadixCache).
Within-run A/B under a bounded KV pool (``--max-total-tokens``):
- Prompt A is requested with a top-level ``session_id``; once the request
finishes, its reusable cache leaves are registered under the session.
- Prompt B is requested without a session. B doubles as the flag-off
control: it is exactly what happens to every prompt without session
protection, so no second server launch is needed.
- Unique flood prompts pressure the pool to ~2x capacity. The unprotected
B must be evicted while the session-referenced A survives (soft
protection).
- ``/close_session`` releases A's references; a second flood round must
then evict A like any unprotected entry.

Manual run on Ascend NPU where weights live outside the CI modelscope cache:
SGLANG_TEST_MODEL_PATH=/home/weights/Llama-3.2-1B-Instruct \
python3 -m unittest test_session_radix_cache_e2e -v
"""

import os
import random
import unittest
import uuid

import requests

from sglang.srt.utils import is_npu, kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci, register_npu_ci
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=400, stage="extra-a", runner_config="1-gpu-small")
register_npu_ci(est_time=500, suite="full-1-npu-a3", nightly=True)

MAX_TOTAL_TOKENS = 8192
WORDS_PER_PROMPT = 1200
NUM_FLOOD_PROMPTS = 10
KEEP_THRESHOLD = 0.75
EVICT_THRESHOLD = 0.20

_WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega quantum vector tensor "
    "matrix gradient manifold topology lattice kernel signal photon proton "
    "neutron electron quark gluon plasma fusion fission orbit comet nebula "
    "galaxy pulsar quasar meteor asteroid eclipse horizon zenith nadir tundra "
    "savanna prairie canyon mesa plateau fjord archipelago lagoon reef river "
    "glacier volcano earthquake tornado hurricane monsoon blizzard"
).split()

QWEN3_0_6B_WEIGHTS_PATH="/home/weights/Qwen3-0.6B"
def _make_prompt(seed: int) -> str:
    rng = random.Random(seed)
    salt = uuid.uuid4().hex
    body = " ".join(rng.choice(_WORDS) for _ in range(WORDS_PER_PROMPT))
    return f"[{salt}] {body}"


class TestSessionRadixCacheE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH

        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--context-length",
            str(MAX_TOTAL_TOKENS),
            "--max-total-tokens",
            str(MAX_TOTAL_TOKENS),
            "--enable-session-radix-cache",
        ]

        # env = {
        #     "SGLANG_ENABLE_UNIFIED_RADIX_TREE": "1",
        # }

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, text, session_id=None):
        payload = {
            "text": text,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        }
        if session_id is not None:
            payload["session_id"] = session_id

        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        meta = response.json()["meta_info"]
        return meta["prompt_tokens"], meta.get("cached_tokens", 0)

    def _cached_ratio(self, text, session_id=None):
        prompt_tokens, cached_tokens = self._generate(text + " Continue.", session_id=session_id)
        self.assertGreater(prompt_tokens, 0)
        return cached_tokens / prompt_tokens

    def test_session_protection_and_release(self):
        session_id = f"e2e-session-{uuid.uuid4().hex[:8]}"
        prompt_a = _make_prompt(seed=1)
        prompt_b = _make_prompt(seed=2)

        a_seed = self._generate(prompt_a, session_id=session_id)
        b_seed = self._generate(prompt_b)

        self.assertEqual(a_seed[1], 0, "first request should have no cache hit")
        self.assertEqual(b_seed[1], 0, "first request should have no cache hit")

        for i in range(NUM_FLOOD_PROMPTS):
            self._generate(_make_prompt(seed=100 + i))

        b_ratio = self._cached_ratio(prompt_b)
        self.assertLess(
            b_ratio,
            EVICT_THRESHOLD,
            f"unprotected prompt B should be evicted, cached_ratio={b_ratio:.3f}",
        )

        a_ratio = self._cached_ratio(prompt_a, session_id=session_id)
        self.assertGreaterEqual(
            a_ratio,
            KEEP_THRESHOLD,
            f"session-protected prompt A should survive, cached_ratio={a_ratio:.3f}",
        )

        response = requests.post(
            f"{self.base_url}/close_session",
            json={"session_id": session_id},
            timeout=60,
        )
        response.raise_for_status()

        for i in range(NUM_FLOOD_PROMPTS):
            self._generate(_make_prompt(seed=200 + i))

        a_ratio_after = self._cached_ratio(prompt_a)
        self.assertLess(
            a_ratio_after,
            EVICT_THRESHOLD,
            "prompt A should be evicted after close_session, "
            f"cached_ratio={a_ratio_after:.3f}",
        )


if __name__ == "__main__":
    unittest.main()
