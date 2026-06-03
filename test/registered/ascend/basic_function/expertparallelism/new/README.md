# PrefillDelayer NPU 测试文档

## 概述

本文档描述 SGLang 中 PrefillDelayer 组件在昇腾 (NPU) 平台上的测试实现。PrefillDelayer 是一个调度优化组件，用于在分布式推理中协调 prefill 阶段的执行时机，以提高吞吐量和资源利用率。

## 测试文件说明

本目录包含三个版本的测试文件：

| 文件名 | 说明 | 状态 |
|-------|------|------|
| `test_prefill_delayer.py` | CUDA GPU (H200) 原版测试 | 上游参考 |
| `_new_test_npu_prefill_delayer.py` | NPU 新版测试（推荐） | **推荐使用** |
| `zz_test_npu_prefill_delayer.py` | NPU 旧版测试（简化版） | 可考虑废弃 |

## 快速开始

### 环境要求

- 昇腾 NPU 设备（8卡）
- Python 3.8+
- SGLang 框架已安装
- 预下载模型权重

### 模型准备

确保以下模型权重已下载到本地：

```python
# 在 sglang/test/ascend/test_ascend_utils.py 中配置
QWEN3_0_6B_WEIGHTS_PATH = "/path/to/qwen3-0.6b"
DEEPSEEK_CODER_V2_LITE_WEIGHTS_PATH = "/path/to/deepseek-coder-v2-lite"
```

### 运行测试

```bash
# 运行所有测试
python -m pytest _new_test_npu_prefill_delayer.py -v

# 运行特定测试类
python -m pytest _new_test_npu_prefill_delayer.py::TestNPUPrefillDelayerNegotiate -v
python -m pytest _new_test_npu_prefill_delayer.py::TestPrefillDelayerThroughputOfflineGen -v

# 运行特定测试方法
python -m pytest _new_test_npu_prefill_delayer.py::TestNPUPrefillDelayerNegotiate::test_negotiate -v
```

### 环境变量

```bash
# 设置测试使用的 NPU 数量（默认 8）
export SGLANG_TEST_WORLD_SIZE=8

# 启用 PrefillDelayer 调试日志
export SGLANG_PREFILL_DELAYER_DEBUG_LOG=1
```

## 测试架构

### 1. 单元测试 (`TestNPUPrefillDelayerNegotiate`)

测试 `PrefillDelayer._negotiate_should_allow_prefill()` 核心协商逻辑。

#### 测试数据结构

```python
@dataclass
class NegotiateCall:
    prefillable: List[bool]              # 各 rank 是否可 prefill
    token_usage: List[float]             # 各 rank token 使用率
    running_batch: Optional[List[int]]   # 各 rank 运行中 batch 数
    max_prefill_bs: Optional[List[int]]  # 各 rank 最大 prefill batch size
    waiting_queue_len: Optional[List[int]]  # 各 rank 等待队列长度
    max_running_requests: Optional[int]  # 最大运行请求数
    sleep_before_s: float = 0.0          # 调用前休眠时间

@dataclass
class NegotiateTestCase:
    name: str                            # 测试用例名称
    max_delay_passes: int                # 最大延迟轮数
    token_usage_low_watermark: Optional[float]  # token 使用率低水位
    calls: List[NegotiateCall]           # 调用序列
    expected_allow: bool                 # 期望是否允许 prefill
    expected_reason: str                 # 期望原因
    queue_min_ratio: Optional[float]     # 队列最小比例（队列触发）
    max_delay_ms: Optional[float]        # 最大延迟毫秒（队列触发）
```

#### 测试用例列表

| 用例名 | 场景描述 | 关键参数 |
|-------|---------|---------|
| `all_prefillable` | 全部 rank 可 prefill | `prefillable=[True,True,True,True]` |
| `all_prefillable_with_previous_wait` | 之前有等待，现在全部可执行 | 两次调用序列 |
| `none_prefillable` | 无 rank 可 prefill | `prefillable=[False,False,False,False]` |
| `mixed_delay` | 部分可 prefill，触发延迟 | `prefillable=[True,False,True,False]` |
| `mixed_watermark_force_allow` | 低 token 使用率强制允许 | `token_usage=[0.5,0.9,0.9,0.9]` |
| `mixed_watermark_disabled` | 水印禁用 | `token_usage_low_watermark=None` |
| `mixed_watermark_not_prefillable` | 不可 prefill 时水印不生效 | `prefillable=[False,False,True,False]` |
| `mixed_timeout` | 超过最大延迟次数后强制允许 | `max_delay_passes=3` |
| `queue_trigger_delay` | 队列低于阈值时延迟 | `queue_min_ratio=0.5`, `waiting_queue_len=[10]` |
| `queue_trigger_above_threshold` | 队列高于阈值时允许 | `waiting_queue_len=[64]` |
| `queue_trigger_disabled_when_ratio_unset` | 未配置时禁用队列触发 | `queue_min_ratio=None` |
| `queue_trigger_wall_clock_timeout` | 队列触发超时机制 | `max_delay_ms=50`, `sleep_before_s=0.2` |

#### 队列触发算法

```python
# queue_min 计算逻辑
queue_min = min(running_batch * queue_min_ratio, max_prefill_bs)

# 示例
running_batch = 100
queue_min_ratio = 0.5
max_prefill_bs = 80
queue_min = min(100 * 0.5, 80) = min(50, 80) = 50

# 触发条件
if waiting_queue_len < queue_min:
    # 延迟 prefill，原因 "delay"
else:
    # 允许 prefill，原因 "no_wait"
```

### 2. 吞吐量测试

#### 2.1 在线服务测试 (`TestNPUPrefillDelayerThroughputOnlineServing`)

**测试目标**：验证在线服务场景下 PrefillDelayer 的功能正确性

**测试参数**：
```python
num_prompts=500              # 请求数量
random_input_len=30000       # 输入长度
random_output_len=256        # 输出长度
request_rate=32              # 请求速率 (req/s)
schedule_policy="lpm"        # LPM 调度策略
```

**启动参数**：
```bash
--schedule-policy lpm
--attention-backend ascend
--disable-cuda-graph
```

**验证内容**：
- 服务器正常启动
- 基准测试完成
- 指标正确上报

#### 2.2 离线生成测试 (`TestPrefillDelayerThroughputOfflineGen`)

**测试目标**：验证离线场景下吞吐量提升 ≥ 20%

**测试参数**：
```python
num_prompts=800              # 请求数量
random_input_len=30000       # 输入长度
random_output_len=500        # 输出长度
max_total_tokens=200000      # 最大总 token 数
token_usage_low_watermark=0.8  # token 使用率低水位
```

**验证逻辑**：
```python
improvement_pct = (enabled - disabled) / disabled * 100
assert improvement_pct >= 20
```

### 3. Token 使用率水印测试 (`TestPrefillDelayerTokenUsageLowWatermark`)

**测试目标**：验证低水印机制对请求优先级的影响

#### 测试方法 1: 启用水印 (`test_1_with_low_watermark`)

```python
token_usage_low_watermark=0.5
max_delay_passes=3000
```

**预期结果**：短请求应在 5 秒内完成

#### 测试方法 2: 禁用水印 (`test_2_without_low_watermark`)

```python
token_usage_low_watermark=None
```

**预期结果**：短请求会被长请求阻塞，耗时 > 5 秒

**注意**：当前被跳过，等待 DP-attention detokenizer hang 问题修复

#### 测试流程

1. 发送长 prompt 阻塞请求（5000 tokens）到 DP rank 0
2. 等待 3 秒后，发送多个短请求到其他 DP rank (1-7)
3. 验证短请求的完成时间

### 4. 精度测试 (`TestPrefillDelayerAccuracy`)

**测试目标**：验证 PrefillDelayer 不影响模型推理精度

**测试数据集**：GSM8K

**准确率要求**：> 57%

**测试方法**：
- `test_1_gsm8k_has_prefill_delayer`：启用 PrefillDelayer
- `test_2_gsm8k_no_prefill_delayer`：禁用 PrefillDelayer

## 核心参数说明

### PrefillDelayer 配置参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|-------|------|
| `--enable-prefill-delayer` | bool | False | 启用 PrefillDelayer |
| `--prefill-delayer-max-delay-passes` | int | 100 | 最大延迟轮数 |
| `--prefill-delayer-token-usage-low-watermark` | float | None | token 使用率低水位 |
| `--prefill-delayer-queue-min-ratio` | float | None | 队列最小比例（队列触发）|
| `--prefill-delayer-max-delay-ms` | float | None | 最大延迟毫秒（队列触发）|

### 服务器启动参数

```bash
--trust-remote-code
--tp 8                          # Tensor Parallelism
--dp 8                          # Data Parallelism
--enable-dp-attention          # 启用 DP Attention
--chunked-prefill-size 131072  # Chunked prefill 大小
--mem-fraction-static 0.6      # 静态内存比例
--attention-backend ascend     # 昇腾注意力后端
--disable-cuda-graph           # 禁用 CUDA graph
--enable-metrics               # 启用指标上报
```

## Prometheus 指标

PrefillDelayer 上报以下指标：

```
sglang:prefill_delayer_wait_forward_passes    # 等待的前向传播次数
sglang:prefill_delayer_wait_seconds           # 等待时间（秒）
sglang:prefill_delayer_outcomes_total         # 各种结果的总数
```

结果类型包括：
- `no_wait`：无需等待
- `wait_success`：等待成功
- `wait_timeout`：等待超时
- `delay`：延迟执行
- `token_watermark`：token 水印触发

## 故障排查

### 常见问题

#### 1. 测试超时

**现象**：`test_negotiate` 或其他测试超时

**解决方案**：
```bash
# 增加超时时间
export DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH=600
```

#### 2. 吞吐量提升不达标

**现象**：`improvement_pct < 20%`

**可能原因**：
- NPU 驱动版本不匹配
- 模型权重加载异常
- 其他进程占用 NPU 资源

**排查步骤**：
```bash
# 检查 NPU 状态
npu-smi info

# 检查进程占用
ps aux | grep python
```

#### 3. 精度测试失败

**现象**：`score <= 0.57`

**可能原因**：
- 模型权重不完整
- 随机种子影响
- 精度计算方式差异

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看 PrefillDelayer 内部状态
print(delayer._state)

# 检查指标
metrics = requests.get(f"{base_url}/metrics").text
print(metrics)
```

## 版本对比

| 特性 | CUDA 原版 | NPU 新版 | NPU 旧版 |
|-----|----------|---------|---------|
| 队列触发机制 | ✅ | ✅ | ❌ |
| 昇腾后端支持 | ❌ | ✅ | ✅ |
| 完整测试用例 | ✅ | ✅ | ❌ |
| 推荐状态 | 上游参考 | **推荐使用** | 可考虑废弃 |

## 参考资料

- [SGLang 官方文档](https://docs.sglang.ai/)
- [昇腾 NPU 开发文档](https://www.hiascend.com/document)
- [PrefillDelayer 设计文档](https://github.com/sgl-project/sglang/issues/XXXX)

## 维护信息

- **维护者**：Ascend NPU 测试团队
- **最后更新**：2026-06-02
- **版本**：v2.0
