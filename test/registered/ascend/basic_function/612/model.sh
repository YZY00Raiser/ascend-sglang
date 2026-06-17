from sglang.test.ascend.test_ascend_utils import GTE_QWEN2_1_5B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V2_LITE_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
QWEN3_0_6B_WEIGHTS_PATH="/home/weights/Qwen/Qwen3-0.6B"

register_npu_ci(est_time=400, suite="full-2-npu-a3", nightly=True)

"--attention-backend",
                "ascend",
                attention_backend="ascend"


    def setUpClass(cls):
        cls.process = popen_launch_server(
            QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[

                "--expert-distribution-recorder-mode",
                cls.expert_distribution_recorder_mode,
            ],
            env={

                "TRANSFORMERS_VERBOSITY": "error",
            },
        )
