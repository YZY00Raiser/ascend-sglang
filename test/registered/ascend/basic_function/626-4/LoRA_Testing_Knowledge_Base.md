# LoRA Testing Knowledge Base / LoRA 测试知识库

## Overview / 概述

This knowledge base covers LoRA (Low-Rank Adaptation) testing for Qwen3 and Qwen3.5 model families in the SGLang framework. The tests validate LoRA adapter integration, logprob accuracy, and cross-platform compatibility (CUDA/Ascend/NPU).

本知识库涵盖 SGLang 框架中 Qwen3 和 Qwen3.5 模型系列的 LoRA（低秩自适应）测试。这些测试验证 LoRA 适配器集成、logprob 精度和跨平台兼容性（CUDA/Ascend/NPU）。

**Supported Models / 支持的模型**:
- Qwen3-4B / Qwen3-8B
- Qwen3-30B-A3B-Instruct / Qwen3-30B-A3B-Instruct-2507
- Qwen3.5-4B / Qwen3.5-35B-A3B
- Qwen3-VL-30B-A3B-Instruct

**Core Testing Dimensions / 核心测试维度**:
- LoRA logprob accuracy comparison against reference training data / LoRA logprob 精度与参考训练数据的对比
- Base model vs LoRA model logprob differentiation / 基础模型与 LoRA 模型 logprob 的差异性
- Cross-platform validation (CUDA, Ascend NPU) / 跨平台验证（CUDA、Ascend NPU）
- Multi-GPU tensor parallelism support / 多 GPU 张量并行支持

## Core Parameters / 核心参数

| Parameter / 参数 | Description / 描述 | Test Coverage / 测试覆盖 |
|------------------|-------------------|-------------------------|
| `--enable_lora` | Enable LoRA adapter support / 启用 LoRA 适配器支持 | ✅ All test files / 所有测试文件 |
| `--max_lora_rank` | Maximum LoRA rank for adapter layers / 适配器层的最大 LoRA 秩 | ✅ All test files / 所有测试文件 |
| `--lora_paths` | Dictionary mapping LoRA names to adapter paths / LoRA 名称到适配器路径的映射字典 | ✅ All test files / 所有测试文件 |
| `--lora_backend` | Backend implementation for LoRA operations (triton) / LoRA 操作的后端实现（triton） | ✅ All test files / 所有测试文件 |
| `--tp_size` | Tensor parallelism size for distributed inference / 分布式推理的张量并行大小 | ✅ All test files / 所有测试文件 |
| `--moe_runner_backend` | MoE runner backend for sparse models / 稀疏模型的 MoE 运行器后端 | ✅ test_lora_qwen3_5_35b_a3b_logprob_diff.py |
| `--experts_shared_outer_loras` | Enable shared experts to use outer LoRAs / 启用共享专家使用外部 LoRA | ✅ test_lora_qwen3_5_35b_a3b_logprob_diff.py |
| `--lora_use_virtual_experts` | Use virtual expert mapping for MoE / 为 MoE 使用虚拟专家映射 | ✅ test_lora_qwen3_5_35b_a3b_logprob_diff.py |
| `--disable_shared_experts_fusion` | Disable fusion of shared experts / 禁用共享专家的融合 | ✅ test_lora_qwen3_5_35b_a3b_logprob_diff.py |
| `--chunked_prefill_size` | Chunked prefill size for memory optimization / 用于内存优化的分块预填充大小 | ✅ test_lora_qwen3_5_35b_a3b_logprob_diff.py |
| `--mem_fraction_static` | Static memory fraction for KV cache / KV 缓存的静态内存比例 | ✅ test_lora_qwen3_5_35b_a3b_logprob_diff.py |
| `--return_logprob` | Return token log probabilities / 返回 token 对数概率 | ✅ All test files / 所有测试文件 |
| `--logprob_start_len` | Starting position for logprob calculation / logprob 计算的起始位置 | ✅ All test files / 所有测试文件 |

## Test Function Points / 测试功能点

### 1. LoRA Logprob Accuracy Test - Qwen3.5-4B 🔬 (test_lora_qwen3_5_4b_logprob_diff.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**: Validate that SGLang LoRA logprobs match reference training logprobs within KL divergence threshold for Qwen3.5-4B model / 验证 SGLang LoRA logprob 与 Qwen3.5-4B 模型的参考训练 logprob 在 KL 散度阈值内匹配

**Test Type / 测试类型**: Precision test / 精度测试

**Covered Parameters / 覆盖参数**:
- `enable_lora`
- `max_lora_rank` (64)
- `lora_paths`
- `lora_backend` (triton)
- `tp_size` (1)
- `return_logprob`
- `logprob_start_len`

**Function Points / 功能点**:
- Download LoRA adapter and reference data from HuggingFace / 从 HuggingFace 下载 LoRA 适配器和参考数据
- Initialize SGLang engine with LoRA support / 使用 LoRA 支持初始化 SGLang 引擎
- Extract prompt logprobs for base model and LoRA model / 提取基础模型和 LoRA 模型的提示 logprob
- Compare base vs LoRA logprobs to ensure difference exists / 比较基础模型与 LoRA 模型的 logprob 以确保存在差异
- Calculate KL divergence between SGLang and training logprobs / 计算 SGLang 与训练 logprob 之间的 KL 散度
- Verify KL divergence is within threshold (4e-3) / 验证 KL 散度在阈值（4e-3）内

**Observable Points / 可观察点**:
- KL(sglang, trainer) <= 4e-3 / KL(sglang, trainer) <= 4e-3
- Base vs LoRA logprob difference (mean_diff, max_diff) / 基础模型与 LoRA 模型 logprob 差异（平均差异、最大差异）
- Logprob tensor equality check (should be False) / logprob 张量相等性检查（应为 False）

---

### 2. LoRA Logprob Accuracy Test - Qwen3-8B 🔬 (test_lora_qwen3_8b_logprob_diff.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**: Validate LoRA logprob accuracy for Qwen3-8B with mock module structure testing / 验证 Qwen3-8B 的 LoRA logprob 精度，包含模拟模块结构测试

**Test Type / 测试类型**: Precision test / 精度测试

**Covered Parameters / 覆盖参数**:
- `enable_lora`
- `max_lora_rank` (32)
- `lora_paths`
- `lora_backend` (triton)
- `tp_size` (1)
- `prefill_attention_backend` (fa4)
- `decode_attention_backend` (fa4)
- `return_logprob`
- `logprob_start_len`

**Function Points / 功能点**:
- Download LoRA adapter from HuggingFace dataset / 从 HuggingFace 数据集下载 LoRA 适配器
- Build mock Qwen3-8B module structure for LoRA target detection / 构建模拟 Qwen3-8B 模块结构用于 LoRA 目标检测
- Test auto_detect_lora_target_modules with mock structure / 使用模拟结构测试 auto_detect_lora_target_modules
- Compare SGLang logprobs against reference training logprobs / 将 SGLang logprob 与参考训练 logprob 进行对比
- Verify KL divergence threshold (5e-3) / 验证 KL 散度阈值（5e-3）

**Observable Points / 可观察点**:
- KL divergence between SGLang and training logprobs / SGLang 与训练 logprob 之间的 KL 散度
- Mock module structure correctness / 模拟模块结构正确性
- LoRA target module auto-detection accuracy / LoRA 目标模块自动检测精度

---

### 3. LoRA Logprob Accuracy Test - Qwen3.5-35B-A3B 🔬 (test_lora_qwen3_5_35b_a3b_logprob_diff.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**: Validate LoRA logprob accuracy for large MoE model Qwen3.5-35B-A3B with 4-GPU tensor parallelism / 验证大型 MoE 模型 Qwen3.5-35B-A3B 在 4-GPU 张量并行下的 LoRA logprob 精度

**Test Type / 测试类型**: Precision test / 精度测试

**Covered Parameters / 覆盖参数**:
- `enable_lora`
- `max_lora_rank` (64)
- `lora_paths`
- `lora_backend` (triton)
- `tp_size` (4)
- `moe_runner_backend` (triton)
- `experts_shared_outer_loras` (True)
- `lora_use_virtual_experts` (True)
- `disable_shared_experts_fusion` (True)
- `chunked_prefill_size` (8192)
- `mem_fraction_static` (0.8)

**Function Points / 功能点**:
- Multi-GPU tensor parallelism (4 GPUs) with LoRA / 多 GPU 张量并行（4 GPU）与 LoRA
- MoE (Mixture of Experts) model LoRA support / MoE（混合专家）模型 LoRA 支持
- Virtual expert mapping for LoRA adapters / LoRA 适配器的虚拟专家映射
- Memory optimization with chunked prefill and static memory fraction / 使用分块预填充和静态内存比例进行内存优化
- KL divergence validation with strict threshold (1e-3) / 严格阈值（1e-3）的 KL 散度验证

**Observable Points / 可观察点**:
- KL(sglang, trainer) <= 1e-3 / KL(sglang, trainer) <= 1e-3
- Multi-GPU LoRA synchronization / 多 GPU LoRA 同步
- Memory usage within configured fraction / 内存使用在配置的范围内
- MoE expert routing with LoRA / 带有 LoRA 的 MoE 专家路由

---

### 4. Basic LoRA Model Test - Qwen3 🔗 (test_lora_qwen3.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**: Validate basic LoRA adapter loading and inference for Qwen3 models using preflight tolerance checks / 验证 Qwen3 模型的基础 LoRA 适配器加载和推理，使用预检容差检查

**Test Type / 测试类型**: Integration test / 集成测试

**Covered Parameters / 覆盖参数**:
- `enable_lora`
- `max_loras_per_batch` (2)
- `prefill_tolerance` (3e-1)

**Function Points / 功能点**:
- Load multiple LoRA adapters per batch / 每批次加载多个 LoRA 适配器
- Prefill tolerance validation for output correctness / 预填充容差验证输出正确性
- Support for multiple adapter sources (nissenj, TanXS) / 支持多个适配器源（nissenj、TanXS）
- AMD CI integration test framework / AMD CI 集成测试框架

**Observable Points / 可观察点**:
- Prefill output tolerance within threshold / 预填充输出容差在阈值内
- Multiple LoRA adapter loading success / 多个 LoRA 适配器加载成功
- Batch inference correctness / 批量推理正确性

---

### 5. NPU-Specific LoRA Tests 🖥️ (test_npu_*.py) [Platform Integration Test / 平台集成测试]

**Test Goal / 测试目标**: Validate LoRA functionality on Ascend NPU hardware for Qwen3 model variants / 验证 Ascend NPU 硬件上 Qwen3 模型变体的 LoRA 功能

**Test Type / 测试类型**: Platform integration test / 平台集成测试

**Covered Files / 覆盖文件**:
- `test_lora_npu_qwen3_5_4b_logprob_diff.py`
- `test_npu_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py`
- `test_npu_lora_qwen3_5_35b_a3b_logprob_diff.py`
- `test_npu_lora_qwen3_8b_logprob_diff.py`
- `test_npu_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py`

**Function Points / 功能点**:
- Ascend NPU hardware compatibility / Ascend NPU 硬件兼容性
- NPU-specific LoRA backend optimization / NPU 特定的 LoRA 后端优化
- Cross-platform logprob consistency / 跨平台 logprob 一致性
- Vision-Language model LoRA support (Qwen3-VL) / 视觉语言模型 LoRA 支持（Qwen3-VL）

**Observable Points / 可观察点**:
- NPU device utilization / NPU 设备利用率
- Logprob accuracy on NPU / NPU 上的 logprob 精度
- Cross-platform result consistency / 跨平台结果一致性

---

### 6. LoRA Logprob Test - Qwen3-VL-30B-A3B-Instruct 🔬 (test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**: Validate LoRA logprob accuracy for Vision-Language MoE model / 验证视觉语言 MoE 模型的 LoRA logprob 精度

**Test Type / 测试类型**: Precision test / 精度测试

**Function Points / 功能点**:
- Vision-Language model LoRA integration / 视觉语言模型 LoRA 集成
- MoE architecture with LoRA adapters / 带有 LoRA 适配器的 MoE 架构
- Multi-modal inference with LoRA / 带有 LoRA 的多模态推理

**Observable Points / 可观察点**:
- KL divergence threshold compliance / KL 散度阈值合规性
- Vision encoder LoRA compatibility / 视觉编码器 LoRA 兼容性

---

### 7. LoRA Logprob Test - Qwen3-30B-A3B-Instruct-2507 🔬 (test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**: Validate LoRA logprob accuracy for Qwen3-30B-A3B-Instruct-2507 variant / 验证 Qwen3-30B-A3B-Instruct-2507 变体的 LoRA logprob 精度

**Test Type / 测试类型**: Precision test / 精度测试

**Function Points / 功能点**:
- Instruction-tuned model LoRA support / 指令微调模型 LoRA 支持
- MoE architecture validation / MoE 架构验证
- Version-specific adapter compatibility (2507) / 特定版本适配器兼容性（2507）

**Observable Points / 可观察点**:
- KL divergence threshold compliance / KL 散度阈值合规性
- Instruction-following quality with LoRA / 带有 LoRA 的指令跟随质量

## Test File Summary / 测试文件汇总

| # | Test File / 测试文件 | Main Function / 主函数 | Test Type / 测试类型 | Category / 类别 |
|---|---------------------|----------------------|---------------------|----------------|
| 1 | test_lora_qwen3_5_4b_logprob_diff.py | test_lora_qwen3_5_4b_logprob_accuracy | Precision / 精度 | Logprob Validation / logprob 验证 |
| 2 | test_lora_qwen3_8b_logprob_diff.py | test_lora_qwen3_8b_logprob_accuracy | Precision / 精度 | Logprob + Mock Testing / logprob + 模拟测试 |
| 3 | test_lora_qwen3_5_35b_a3b_logprob_diff.py | test_lora_qwen3_5_35b_a3b_logprob_accuracy | Precision / 精度 | MoE + Multi-GPU / MoE + 多 GPU |
| 4 | test_lora_qwen3.py | test_ci_lora_models | Integration / 集成 | Basic LoRA / 基础 LoRA |
| 5 | test_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py | test_lora_qwen3_30b_a3b_instruct_2507_logprob_accuracy | Precision / 精度 | Instruction Model / 指令模型 |
| 6 | test_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py | test_lora_qwen3_vl_30b_a3b_instruct_logprob_accuracy | Precision / 精度 | Vision-Language / 视觉语言 |
| 7 | test_lora_npu_qwen3_5_4b_logprob_diff.py | N/A | Platform / 平台 | NPU Integration / NPU 集成 |
| 8 | test_npu_lora_qwen3_30b_a3b_instruct_2507_logprob_diff.py | N/A | Platform / 平台 | NPU + MoE |
| 9 | test_npu_lora_qwen3_5_35b_a3b_logprob_diff.py | N/A | Platform / 平台 | NPU + Large MoE |
| 10 | test_npu_lora_qwen3_8b_logprob_diff.py | N/A | Platform / 平台 | NPU + Dense |
| 11 | test_npu_lora_qwen3_vl_30b_a3b_instruct_logprob_diff.py | N/A | Platform / 平台 | NPU + Vision-Language |

## Observable Points Summary / 可观察点汇总

### Server-side Observables / 服务端可观察点
- Engine initialization with LoRA support / 带有 LoRA 支持的引擎初始化
- LoRA adapter download and loading / LoRA 适配器下载和加载
- Multi-GPU tensor parallelism setup / 多 GPU 张量并行设置
- Memory allocation and KV cache configuration / 内存分配和 KV 缓存配置

### Inference Observables / 推理可观察点
- Prompt logprob extraction accuracy / 提示 logprob 提取精度
- Base vs LoRA logprob differentiation / 基础模型与 LoRA 模型 logprob 差异
- KL divergence calculation correctness / KL 散度计算正确性
- Temperature=0.0 deterministic output / temperature=0.0 的确定性输出

### Performance Observables / 性能可观察点
- Multi-GPU inference throughput / 多 GPU 推理吞吐量
- MoE expert routing latency / MoE 专家路由延迟
- Memory usage with chunked prefill / 使用分块预填充的内存使用
- LoRA adapter switching overhead / LoRA 适配器切换开销

### Error Observables / 错误可观察点
- OOM (Out of Memory) handling / OOM（内存不足）处理
- LoRA adapter loading failures / LoRA 适配器加载失败
- KL divergence threshold violations / KL 散度阈值违规
- Cross-platform consistency errors / 跨平台一致性错误

## KL Divergence Thresholds Summary / KL 散度阈值汇总

| Model / 模型 | KL Threshold / KL 阈值 | Strictness / 严格度 |
|-------------|----------------------|-------------------|
| Qwen3.5-4B | 4e-3 | Medium / 中等 |
| Qwen3-8B | 5e-3 | Medium / 中等 |
| Qwen3.5-35B-A3B | 1e-3 | Strict / 严格 |

## CI Configuration Summary / CI 配置汇总

| Test File / 测试文件 | CI Stage / CI 阶段 | Runner Config / 运行器配置 | Est. Time / 估计时间 |
|---------------------|-------------------|-------------------------|-------------------|
| test_lora_qwen3_5_4b_logprob_diff.py | extra-a | 1-gpu-large | 90s |
| test_lora_qwen3_8b_logprob_diff.py | extra-a | 1-gpu-large | 40s |
| test_lora_qwen3_5_35b_a3b_logprob_diff.py | base-c | 4-gpu-b200 | 110s |
| test_lora_qwen3.py | stage-b-test-1-gpu-small-amd | N/A | 30s |

## Key Implementation Details / 关键实现细节

### KL Divergence Calculation / KL 散度计算
```python
def kl_v2(a, b):
    return (((a - b) ** 2) * 0.5).mean().item()
```
This uses a squared-difference based metric (not true KL divergence) for comparing logprob distributions / 使用基于平方差的度量（非真正的 KL 散度）来比较 logprob 分布。

### Logprob Extraction Pattern / logprob 提取模式
```python
out = engine.generate(
    input_ids=input_ids,
    sampling_params={"max_new_tokens": 0, "temperature": 0.0},
    return_logprob=True,
    logprob_start_len=0,
    lora_path=lora_path,
)
logprobs = [logprob for logprob, _, _ in out["meta_info"]["input_token_logprobs"]][1:]
```
Extracts input token logprobs, skipping the first token / 提取输入 token logprob，跳过第一个 token。

### Engine Configuration Pattern / 引擎配置模式
```python
engine = sgl.Engine(
    model_path=BASE_MODEL,
    tp_size=TP_SIZE,
    enable_lora=True,
    max_lora_rank=MAX_LORA_RANK,
    lora_paths={"my_lora": adapter_path},
    lora_backend=LORA_BACKEND,
)
```
Standard LoRA engine initialization with configurable parameters / 带有可配置参数的标准 LoRA 引擎初始化。
