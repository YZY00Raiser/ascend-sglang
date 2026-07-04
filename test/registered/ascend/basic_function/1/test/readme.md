# 文件说明

qwen3_30b_a3b_random_no_redundant.json 用作字符串格式的输入测试，可以在日志中看到实际的专家分布与我们提供的文件内容一直，精度没问题，说明配置生效了。反向来说你修改其中一个数字，推理应该会报错，也说明我们我们的配置生效了。

expert_distribution_recorder.pt 用作pt配置，没什么说的。

expert_distribution_recorder.json 是直接用pt文件转化过来的，和pt一样的配置方式，也没什么可说的。

# 脚本

## qwen3_30b_a3b_random_no_redundant.json 测试脚本，主要是参数，路径什么改成自己的。

```python
import unittest
import json

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH = "/home/weights/Qwen3-30B-A3B"

# 读取专家分布文件内容，作为参数值
with open("/home/d30060301/pt/qwen3_30b_a3b_random_no_redundant.json", "r") as f:
    init_expert_location = f.read()

class TestQwen330B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the /home/weights/Qwen3-30B-A3B model on the GSM8K dataset is no less than 0.90.

    [Test Category] Model
    [Test Target] /home/weights/Qwen3-30B-A3B
    """

    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        #"--enable-metrics",
        #"--enable-expert-distribution-metrics",
        "--trust-remote-code",
        "--mem-fraction-static",
        0.7,
        "--expert-distribution-recorder-mode",
        "stat",
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
        "--ep-dispatch-algorithm",
        "static",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--init-expert-location",
        init_expert_location   # 直接使用文件内容，而非文件路径
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
        #"SGLANG_EPLB_HEATMAP_COLLECTION_INTERVAL": "100",
        "SGLANG_LOG_EXPERT_LOCATION_METADATA": "1"  # 可以从打屏看到更多信息，用不到的话可以删掉
    }


if __name__ == "__main__":
    unittest.main()
```

## 其他两个的测试脚本，可以看到与原版的脚本相比多了些参数，主要是因为生成pt文件的脚本有这些参数，保持一致，否则报错。

```python
import unittest

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH="/home/weights/Qwen3-30B-A3B"


class TestQwen330B(GSM8KAscendMixin, CustomTestCase):
    """Testcase: Verify that the inference accuracy of the /home/weights/Qwen3-30B-A3B model on the GSM8K dataset is no less than 0.90.

    [Test Category] Model
    [Test Target] /home/weights/Qwen3-30B-A3B
    """

    model = QWEN3_30B_A3B_INSTRUCT_2507_WEIGHTS_PATH
    accuracy = 0.90
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        0.7,
        "--max-running-requests",
        32,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--cuda-graph-max-bs",
        32,
        "--tp-size",
        2,
        "--ep-dispatch-algorithm",
        "static",
        "--moe-a2a-backend",
        "deepep",
        "--deepep-mode",
        "normal",
        "--init-expert-location",
        #"/tmp/pt/expert_distribution_recorder_1783072007.920626.pt",
        #"/home/d30060301/pt/expert_distribution_recorder.pt"
        "/home/d30060301/pt/expert_distribution_recorder.json"
    ]

    env = {
        "HCCL_BUFFSIZE": "1024",
    }


if __name__ == "__main__":
    unittest.main()
```