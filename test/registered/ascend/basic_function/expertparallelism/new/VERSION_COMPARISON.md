# PrefillDelayer 三版本对比文档

## 文件信息

| 属性 | 说明 |
|-----|------|
| **文档版本** | v1.0 |
| **生成日期** | 2026-06-02 |
| **对比文件** | `test_prefill_delayer.py` (CUDA原版) / `_new_test_npu_prefill_delayer.py` (NPU新版) / `zz_test_npu_prefill_delayer.py` (NPU旧版) |

---

## 一、基础信息对比

### 1.1 文件元数据

| 项目 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| **文件名** | `test_prefill_delayer.py` | `_new_test_npu_prefill_delayer.py` | `zz_test_npu_prefill_delayer.py` |
| **文件前缀** | 无 | `_new_` | `zz_` |
| **平台** | CUDA (NVIDIA GPU) | NPU (昇腾) | NPU (昇腾) |
| **推荐状态** | 上游参考 | **推荐使用** | 可考虑废弃 |
| **总行数** | ~696 行 | ~728 行 | ~582 行 |

### 1.2 CI 注册信息对比

```python
# CUDA 原版
register_cuda_ci(
    est_time=300,
    stage="base-c",
    runner_config="8-gpu-h200",
    disabled="Temporarily disabled",
)

# NPU 新版
register_npu_ci(
    est_time=400,
    suite="nightly-8-npu-a3",
    nightly=True,
)

# NPU 旧版
register_npu_ci(
    est_time=400,
    suite="nightly-8-npu-a3",
    nightly=True,
)
```

| 属性 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| **注册函数** | `register_cuda_ci` | `register_npu_ci` | `register_npu_ci` |
| **估计时间** | 300 秒 | 400 秒 | 400 秒 |
| **测试套件** | `stage="base-c"` | `suite="nightly-8-npu-a3"` | `suite="nightly-8-npu-a3"` |
| **运行器** | `8-gpu-h200` | `nightly-8-npu-a3` | `nightly-8-npu-a3` |
| **禁用状态** | `Temporarily disabled` | 无 | 无 |

---

## 二、导入模块对比

### 2.1 模型路径导入

```python
# CUDA 原版 - 使用 HuggingFace 在线模型
model = "Qwen/Qwen3-0.6B"
model = DEFAULT_MLA_MODEL_NAME_FOR_TEST

# NPU 版本 - 使用本地预下载权重
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH,
    QWEN3_0_6B_WEIGHTS_PATH,
)
```

### 2.2 导入差异汇总

| 导入项 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-------|----------|---------|---------|
| `register_cuda_ci` | ✅ | ❌ | ❌ |
| `register_npu_ci` | ❌ | ✅ | ✅ |
| `DEFAULT_MLA_MODEL_NAME_FOR_TEST` | ✅ | ❌ | ❌ |
| `DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH` | ❌ | ✅ | ✅ |
| `QWEN3_0_6B_WEIGHTS_PATH` | ❌ | ✅ | ✅ |

---

## 三、数据结构对比

### 3.1 NegotiateCall 类对比

```python
# CUDA 原版 & NPU 新版（完整版）
@dataclass
class NegotiateCall:
    prefillable: List[bool]
    token_usage: List[float]
    running_batch: Optional[List[int]] = None      # 队列触发
    max_prefill_bs: Optional[List[int]] = None     # 队列触发
    waiting_queue_len: Optional[List[int]] = None  # 队列触发
    max_running_requests: Optional[int] = None     # 队列触发
    sleep_before_s: float = 0.0                    # 超时测试

# NPU 旧版（简化版）
@dataclass
class NegotiateCall:
    prefillable: List[bool]
    token_usage: List[float]
    # 无队列触发相关字段
```

### 3.2 NegotiateTestCase 类对比

```python
# CUDA 原版 & NPU 新版（完整版）
@dataclass
class NegotiateTestCase:
    name: str
    max_delay_passes: int
    token_usage_low_watermark: Optional[float]
    calls: List[NegotiateCall]
    expected_allow: bool
    expected_reason: str
    queue_min_ratio: Optional[float] = None    # 队列触发
    max_delay_ms: Optional[float] = None       # 队列触发

# NPU 旧版（简化版）
@dataclass
class NegotiateTestCase:
    name: str
    max_delay_passes: int
    token_usage_low_watermark: Optional[float]
    calls: List[NegotiateCall]
    expected_allow: bool
    expected_reason: str
    # 无 queue_min_ratio 和 max_delay_ms
```

### 3.3 数据结构差异总结

| 字段 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| `running_batch` | ✅ | ✅ | ❌ |
| `max_prefill_bs` | ✅ | ✅ | ❌ |
| `waiting_queue_len` | ✅ | ✅ | ❌ |
| `max_running_requests` | ✅ | ✅ | ❌ |
| `sleep_before_s` | ✅ | ✅ | ❌ |
| `queue_min_ratio` | ✅ | ✅ | ❌ |
| `max_delay_ms` | ✅ | ✅ | ❌ |

---

## 四、单元测试用例对比

### 4.1 测试用例数量

| 版本 | 基础用例 | 队列触发用例 | 总计 |
|-----|---------|-------------|------|
| CUDA 原版 | 8 | 4 | **12** |
| NPU 新版 | 8 | 4 | **12** |
| NPU 旧版 | 8 | 0 | **8** |

### 4.2 基础测试用例（三个版本共有）

| # | 用例名 | 描述 | 预期结果 |
|---|-------|------|---------|
| 1 | `all_prefillable` | 全部 rank 可 prefill | `allow=True`, `reason="no_wait"` |
| 2 | `all_prefillable_with_previous_wait` | 之前有等待，现在全部可执行 | `allow=True`, `reason="wait_success"` |
| 3 | `none_prefillable` | 无 rank 可 prefill | `allow=True`, `reason=""` |
| 4 | `mixed_delay` | 部分可 prefill | `allow=False`, `reason="delay"` |
| 5 | `mixed_watermark_force_allow` | 低 token 使用率强制允许 | `allow=True`, `reason="token_watermark"` |
| 6 | `mixed_watermark_disabled` | 水印禁用 | `allow=False`, `reason="delay"` |
| 7 | `mixed_watermark_not_prefillable` | 不可 prefill 时水印不生效 | `allow=False`, `reason="delay"` |
| 8 | `mixed_timeout` | 超过最大延迟次数 | `allow=True`, `reason="wait_timeout"` |

### 4.3 队列触发测试用例（CUDA & NPU 新版独有）

| # | 用例名 | 描述 | 关键参数 |
|---|-------|------|---------|
| 9 | `queue_trigger_delay` | 队列低于阈值时延迟 | `queue_min_ratio=0.5`, `waiting_queue_len=[10]` |
| 10 | `queue_trigger_above_threshold` | 队列高于阈值时允许 | `waiting_queue_len=[64]` |
| 11 | `queue_trigger_disabled_when_ratio_unset` | 未配置时禁用 | `queue_min_ratio=None` |
| 12 | `queue_trigger_wall_clock_timeout` | 队列触发超时 | `max_delay_ms=50`, `sleep_before_s=0.2` |

### 4.4 队列触发算法详解

```python
# queue_min 计算逻辑
queue_min = min(running_batch * queue_min_ratio, max_prefill_bs)

# 调用示例
NegotiateTestCase(
    name="queue_trigger_delay",
    queue_min_ratio=0.5,
    max_delay_ms=5000,
    calls=[
        NegotiateCall(
            prefillable=[True, True, True, True],
            token_usage=[0.9, 0.9, 0.9, 0.9],
            running_batch=[100, 100, 100, 100],
            max_prefill_bs=[80, 80, 80, 80],
            waiting_queue_len=[10, 10, 10, 10],  # 10 < 50 (queue_min)
            max_running_requests=1024,
        ),
    ],
    expected_allow=False,
    expected_reason="delay",
)
```

---

## 五、E2E 吞吐量测试对比

### 5.1 在线服务测试类对比

```python
# CUDA 原版
class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    """Testcase: Online serving scenario...
    [Test Category] Parameter
    [Test Target] --enable-prefill-delayer
    """

# NPU 新版
class TestNPUPrefillDelayerThroughputOnlineServing(CustomTestCase):
    """Test PrefillDelayer throughput for online serving on NPU.
    [Test Category] Scheduler
    [Test Target] PrefillDelayer online serving functionality...
    """

# NPU 旧版
class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    """Testcase: Online serving scenario: Verify that throughput is 
    improved by at least 5% when PrefillDelayer is enabled...
    [Test Category] Parameter
    [Test Target] --enable-prefill-delayer
    """
```

### 5.2 在线服务测试参数对比

| 参数 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| **类名** | `TestPrefillDelayerThroughputOnlineServing` | `TestNPUPrefillDelayerThroughputOnlineServing` | `TestPrefillDelayerThroughputOnlineServing` |
| **测试类别** | `Parameter` | `Scheduler` | `Parameter` |
| **最小提升要求** | `None` | `None` | `5%` |
| `num_prompts` | 500 | 500 | 500 |
| `random_input_len` | 30000 | 30000 | 30000 |
| `random_output_len` | 256 | 256 | 256 |
| `request_rate` | 32 | 32 | 32 |
| `schedule_policy` | `lpm` | `lpm` | `lpm` |
| `--attention-backend` | ❌ | ❌ | `ascend` |
| `--disable-cuda-graph` | ❌ | ❌ | ✅ |

### 5.3 离线生成测试类对比

```python
# CUDA 原版
class TestPrefillDelayerThroughputOfflineGen(CustomTestCase):
    def test_throughput_comparison(self):
        _run_throughput_comparison(
            min_improvement_pct=20,
            token_usage_low_watermark=0.8,
        )

# NPU 新版
class TestPrefillDelayerThroughputOfflineGen(CustomTestCase):
    """Test PrefillDelayer throughput improvement for offline generation on NPU.
    [Test Category] Scheduler
    [Test Target] PrefillDelayer throughput improvement...
    """
    def test_throughput_comparison(self):
        _run_throughput_comparison(
            min_improvement_pct=20,
            token_usage_low_watermark=0.8,
        )

# NPU 旧版
class TestPrefillDelayerThroughputOfflineGen(CustomTestCase):
    """Testcase: Offline generation scenario: Verify that throughput 
    is improved by at least 20%...
    [Test Category] Parameter
    [Test Target] --enable-prefill-delayer; --prefill-delayer-token-usage-low-watermark
    """
    def test_throughput_comparison(self):
        _run_throughput_comparison(
            min_improvement_pct=20,
            token_usage_low_watermark=0.8,
        )
```

### 5.4 离线生成测试参数对比

| 参数 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| **最小提升要求** | `20%` | `20%` | `20%` |
| `num_prompts` | 800 | 800 | 800 |
| `random_input_len` | 30000 | 30000 | 30000 |
| `random_output_len` | 500 | 500 | 500 |
| `max_total_tokens` | 200000 | 200000 | 200000 |
| `token_usage_low_watermark` | 0.8 | 0.8 | 0.8 |
| `--attention-backend` | ❌ | ❌ | `ascend` |
| `--disable-cuda-graph` | ❌ | ❌ | ✅ |

### 5.5 吞吐量对比函数签名差异

```python
# CUDA 原版 & NPU 新版
# 支持功能验证模式（min_improvement_pct 可为 None）
def _run_throughput_comparison(
    test_case,
    test_name: str,
    other_launch_args,
    other_benchmark_args,
    min_improvement_pct: Optional[float],  # Optional
    token_usage_low_watermark: float = None,
):

def _assert_throughput_improvement(
    ...,
    min_improvement_pct: Optional[float],
):
    if min_improvement_pct is None:
        return  # 跳过性能断言

# NPU 旧版
# 始终要求性能断言
def _run_throughput_comparison(
    test_case,
    test_name: str,
    other_launch_args,
    other_benchmark_args,
    min_improvement_pct: float,  # 必须提供
    token_usage_low_watermark: float = None,
):

def _assert_throughput_improvement(
    ...,
    min_improvement_pct: float,  # 必须提供
):
    # 始终执行性能断言，无提前返回
```

---

## 六、Token 使用率水印测试对比

### 6.1 测试类定义对比

```python
# CUDA 原版
class TestPrefillDelayerTokenUsageLowWatermark(CustomTestCase):
    def test_1_with_low_watermark(self):
        self._run(token_usage_low_watermark=0.5)

    @unittest.skip("blocked by sgl-project/sglang#22511")
    def test_2_without_low_watermark(self):
        self._run(token_usage_low_watermark=None)

# NPU 新版
class TestPrefillDelayerTokenUsageLowWatermark(CustomTestCase):
    """Test PrefillDelayer token usage low watermark functionality on NPU.
    [Test Category] Scheduler
    [Test Target] PrefillDelayer token watermark...
    """
    def test_1_with_low_watermark(self):
        self._run(token_usage_low_watermark=0.5)

    @unittest.skip("blocked by sgl-project/sglang#22511")
    def test_2_without_low_watermark(self):
        self._run(token_usage_low_watermark=None)

# NPU 旧版
class TestPrefillDelayerTokenUsageLowWatermark(CustomTestCase):
    """Testcase: Verify PrefillDelayer memory low watermark protection...
    [Test Category] Parameter
    [Test Target] --enable-prefill-delayer; --prefill-delayer-max-delay-passes...
    """
    def test_1_with_low_watermark(self):
        self._run(token_usage_low_watermark=0.5)

    def test_2_without_low_watermark(self):  # 无 skip 装饰器
        self._run(token_usage_low_watermark=None)
```

### 6.2 水印测试参数对比

| 参数 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| **测试类别** | 无 | `Scheduler` | `Parameter` |
| **test_1 水印值** | 0.5 | 0.5 | 0.5 |
| **test_2 状态** | `@unittest.skip` | `@unittest.skip` | 正常运行 |
| **max_delay_passes** | 3000 | 3000 | 100 |
| **启动超时** | 默认 | 默认 | 6000 |
| `--max-total-tokens` | 50000 | 50000 | 50000 |
| `--attention-backend` | ❌ | ❌ | `ascend` |
| `--disable-cuda-graph` | ❌ | ❌ | ✅ |

### 6.3 _run 方法差异

```python
# CUDA 原版 & NPU 新版
process = _launch_server(
    model=model,
    base_url=base_url,
    prefill_delayer=True,
    other_args=["--max-total-tokens", "50000"],
    max_delay_passes=3000,
    token_usage_low_watermark=token_usage_low_watermark,
)

# NPU 旧版
process = _launch_server(
    model=model,
    base_url=base_url,
    prefill_delayer=True,
    other_args=[
        "--max-total-tokens", "50000",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
    ],
    max_delay_passes=100,  # 不同
    token_usage_low_watermark=token_usage_low_watermark,
    timeout=6000,  # 额外参数
)
```

### 6.4 断言错误信息对比

```python
# CUDA 原版 & NPU 新版
f"DP rank {dp_rank} req {req_idx}: elapsed={elapsed:.2f}s, thresh={thresh}, "
f"enabled={enabled}. Maybe you need a different `max_delay_passes` "
f"when using hardware other than H200."

# NPU 旧版
f"DP rank {dp_rank} req {req_idx}: elapsed={elapsed:.2f}s, thresh={thresh}, "
f"enabled={enabled}. You may need a different `max_delay_passes` "
f"on non-H200 hardware."
```

---

## 七、精度测试对比

### 7.1 测试类定义对比

```python
# CUDA 原版
class TestPrefillDelayerAccuracy(CustomTestCase):
    def test_1_gsm8k_has_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=True)

    def test_2_gsm8k_no_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=False)

# NPU 新版
class TestPrefillDelayerAccuracy(CustomTestCase):
    """Test PrefillDelayer accuracy on NPU.
    [Test Category] Scheduler
    [Test Target] PrefillDelayer GSM8K accuracy...
    """
    def test_1_gsm8k_has_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=True)

    def test_2_gsm8k_no_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=False)

# NPU 旧版
class TestPrefillDelayerAccuracy(CustomTestCase):
    """Testcase: Verify that model accuracy on mgsm_en dataset ≥ 87%...
    [Test Category] Parameter
    [Test Target] --enable-prefill-delayer
    """
    def test_1_mgsm_en_has_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=True)

    def test_2_mgsm_en_no_prefill_delayer(self):
        self._run_accuracy_test(prefill_delayer=False)
```

### 7.2 精度测试参数对比

| 参数 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| **测试类别** | 无 | `Scheduler` | `Parameter` |
| **数据集** | `gsm8k` | `gsm8k` | `mgsm_en` |
| **准确率要求** | `> 0.57` (57%) | `> 0.57` (57%) | `> 0.87` (87%) |
| **test_1 方法名** | `test_1_gsm8k_has_prefill_delayer` | `test_1_gsm8k_has_prefill_delayer` | `test_1_mgsm_en_has_prefill_delayer` |
| **test_2 方法名** | `test_2_gsm8k_no_prefill_delayer` | `test_2_gsm8k_no_prefill_delayer` | `test_2_mgsm_en_no_prefill_delayer` |
| **模型** | `DEFAULT_MLA_MODEL_NAME_FOR_TEST` | `DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH` | `DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH` |
| `--max-total-tokens` | 4096 | 4096 | 4096 |
| `--schedule-policy` | `lpm` | `lpm` | `lpm` |
| `--attention-backend` | ❌ | ❌ | `ascend` |
| `--disable-cuda-graph` | ❌ | ❌ | ✅ |

---

## 八、服务器启动函数对比

### 8.1 函数签名对比

```python
# CUDA 原版 & NPU 新版
def _launch_server(
    *,
    model,
    base_url,
    prefill_delayer: bool,
    other_args,
    max_delay_passes: int = 100,
    token_usage_low_watermark: float = None,
):

# NPU 旧版
def _launch_server(
    *,
    model,
    base_url,
    prefill_delayer: bool,
    other_args,
    max_delay_passes: int = 100,
    token_usage_low_watermark: float = None,
    timeout: int = DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,  # 额外参数
):
```

### 8.2 启动参数对比

| 参数 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| `--trust-remote-code` | ✅ | ✅ | ✅ |
| `--tp` | `WORLD_SIZE` | `WORLD_SIZE` | `WORLD_SIZE` |
| `--dp` | `WORLD_SIZE` | `WORLD_SIZE` | `WORLD_SIZE` |
| `--enable-dp-attention` | ✅ | ✅ | ✅ |
| `--chunked-prefill-size` | `131072` | `131072` | `131072` |
| `--mem-fraction-static` | `0.6` | `0.6` | `0.6` |
| `--enable-metrics` | ✅ | ✅ | ✅ |
| `--attention-backend` | ❌ | ❌ | `ascend` |
| `--disable-cuda-graph` | ❌ | ❌ | ✅ |
| `--enable-prefill-delayer` | 条件 | 条件 | 条件 |
| `--prefill-delayer-max-delay-passes` | ✅ | ✅ | ✅ |
| `--prefill-delayer-token-usage-low-watermark` | 条件 | 条件 | 条件 |

---

## 九、代码风格与注释对比

### 9.1 文档字符串风格

```python
# CUDA 原版 - 简洁风格
class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    """Testcase: Online serving scenario: Verify that throughput...
    [Test Category] Parameter
    [Test Target] --enable-prefill-delayer
    """

# NPU 新版 - 详细风格 + NPU 前缀
class TestNPUPrefillDelayerThroughputOnlineServing(CustomTestCase):
    """Test PrefillDelayer throughput for online serving on NPU.
    [Test Category] Scheduler
    [Test Target] PrefillDelayer online serving functionality, 
                  server boot, benchmark completion, metrics emission
    """

# NPU 旧版 - 结构化风格
class TestPrefillDelayerThroughputOnlineServing(CustomTestCase):
    """Testcase: Online serving scenario: Verify that throughput 
    is improved by at least 5% when PrefillDelayer is enabled...
    [Test Category] Parameter
    [Test Target] --enable-prefill-delayer
    """
```

### 9.2 类命名约定

| 测试类 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-------|----------|---------|---------|
| 协商测试 | `TestPrefillDelayerNegotiate` | `TestNPUPrefillDelayerNegotiate` | `TestPrefillDelayerNegotiate` |
| 在线吞吐量 | `TestPrefillDelayerThroughputOnlineServing` | `TestNPUPrefillDelayerThroughputOnlineServing` | `TestPrefillDelayerThroughputOnlineServing` |
| 离线吞吐量 | `TestPrefillDelayerThroughputOfflineGen` | `TestPrefillDelayerThroughputOfflineGen` | `TestPrefillDelayerThroughputOfflineGen` |
| 水印测试 | `TestPrefillDelayerTokenUsageLowWatermark` | `TestPrefillDelayerTokenUsageLowWatermark` | `TestPrefillDelayerTokenUsageLowWatermark` |
| 精度测试 | `TestPrefillDelayerAccuracy` | `TestPrefillDelayerAccuracy` | `TestPrefillDelayerAccuracy` |

---

## 十、演进关系分析

### 10.1 版本演进路径

```
CUDA 原版 (test_prefill_delayer.py)
    │
    ├── 上游参考，保持同步
    │
    ↓
NPU 旧版 (zz_test_npu_prefill_delayer.py)
    │
    ├── 添加昇腾后端支持 (--attention-backend ascend)
    ├── 添加 CUDA Graph 禁用 (--disable-cuda-graph)
    ├── 使用本地模型路径 (非 HuggingFace)
    ├── 简化单元测试（移除队列触发机制）
    ├── 调整精度测试数据集 (gsm8k → mgsm_en)
    ├── 提高精度要求 (57% → 87%)
    ├── 在线服务提升要求 (None → 5%)
    └── 水印测试 test_2 不跳过
    │
    ↓
NPU 新版 (_new_test_npu_prefill_delayer.py)
    │
    ├── 恢复队列触发机制（完整功能）
    ├── 恢复 gsm8k 数据集
    ├── 恢复 57% 精度要求
    ├── 恢复在线服务功能验证模式 (None)
    ├── 添加详细测试类别/目标注释
    ├── 类名添加 NPU 前缀
    └── 水印测试 test_2 添加 skip 装饰器
```

### 10.2 各版本适用场景

| 版本 | 适用场景 | 推荐理由 |
|-----|---------|---------|
| **CUDA 原版** | NVIDIA GPU (H200) 标准测试 | 上游 CI 参考，功能最完整 |
| **NPU 新版** | 昇腾 NPU 完整功能验证 | **推荐使用**，功能完整，注释详细 |
| **NPU 旧版** | 昇腾 NPU 基础功能验证 | 简化版本，无需队列触发机制时使用 |

---

## 十一、关键差异总结

### 11.1 功能特性对比表

| 特性 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| 队列触发机制 | ✅ | ✅ | ❌ |
| 昇腾后端支持 | ❌ | ✅ | ✅ |
| 完整测试用例 | ✅ | ✅ | ❌ |
| 详细注释文档 | ❌ | ✅ | ✅ |
| 在线服务性能断言 | ❌ | ❌ | ✅ |
| 水印测试 test_2 跳过 | ✅ | ✅ | ❌ |
| 类名 NPU 前缀 | ❌ | ✅ | ❌ |
| HuggingFace 模型 | ✅ | ❌ | ❌ |
| 本地模型路径 | ❌ | ✅ | ✅ |

### 11.2 数值参数差异汇总

| 参数 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| 精度要求 | 57% | 57% | 87% |
| 在线服务提升 | None | None | 5% |
| 离线生成提升 | 20% | 20% | 20% |
| 水印测试 max_delay_passes | 3000 | 3000 | 100 |
| 水印测试超时 | 默认 | 默认 | 6000 |
| CI 估计时间 | 300s | 400s | 400s |

---

## 十二、维护建议

### 12.1 版本管理建议

1. **NPU 旧版 (zz_)** 
   - 状态：可考虑废弃
   - 原因：功能已被新版完全覆盖
   - 建议：归档或删除

2. **NPU 新版 (_new_)**
   - 状态：**推荐使用**
   - 原因：功能完整，注释详细
   - 建议：作为昇腾平台标准测试文件

3. **CUDA 原版**
   - 状态：上游参考
   - 原因：保持与上游同步
   - 建议：定期同步上游更新到 NPU 版本

### 12.2 代码合并建议

考虑将三个版本统一为一个文件，通过条件编译或配置参数区分平台：

```python
import platform

if platform.system() == "Linux" and is_npu_available():
    # NPU 配置
    ATTENTION_BACKEND = "ascend"
    DISABLE_CUDA_GRAPH = True
    MODEL_PATH = QWEN3_0_6B_WEIGHTS_PATH
else:
    # CUDA 配置
    ATTENTION_BACKEND = None
    DISABLE_CUDA_GRAPH = False
    MODEL_PATH = "Qwen/Qwen3-0.6B"
```

### 12.3 同步策略

| 方向 | 频率 | 内容 |
|-----|------|------|
| CUDA → NPU 新版 | 每版本发布 | 新功能、bug 修复 |
| NPU 新版 → NPU 旧版 | 不再同步 | 旧版进入维护模式 |
| NPU → CUDA | 按需 | NPU 特定优化反馈上游 |

---

## 附录：文件哈希值

| 文件 | MD5 (示例) |
|-----|-----------|
| `test_prefill_delayer.py` | `a1b2c3d4e5f6...` |
| `_new_test_npu_prefill_delayer.py` | `b2c3d4e5f6g7...` |
| `zz_test_npu_prefill_delayer.py` | `c3d4e5f6g7h8...` |

---

**文档结束**
