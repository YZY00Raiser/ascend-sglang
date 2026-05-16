# Mamba Layers Testing Knowledge Base / Mamba 层测试知识库

## Overview / 概述

This document provides a comprehensive overview of the test suite for Mamba layers in SGLang. Mamba is a state space model (SSM) architecture that provides efficient sequence modeling with linear complexity relative to sequence length.

本文档提供了 SGLang 中 Mamba 层测试套件的全面概述。Mamba 是一种状态空间模型（SSM）架构，能够以相对于序列长度的线性复杂度提供高效的序列建模。

### Supported Architectures / 支持的架构
- CUDA GPUs (NVIDIA)
- XPU devices (Intel)
- AMD GPUs (ROCm/HIP)

### Core Testing Dimensions / 核心测试维度
1. **Causal Convolution (1D)** - Causal convolution operations for temporal modeling
2. **Selective State Update (SSU)** - State update mechanisms with selective parameters
3. **Mamba Chunk Scan** - Chunked scan operations for efficient SSM computation
4. **Mamba2 Mixer** - Multi-GPU tensor parallel RMS norm gated mixer

---

## Core Parameters / 核心参数

| Parameter / 参数 | Description / 描述 | Test Coverage / 测试覆盖 |
|------------------|-------------------|-------------------------|
| `itype` | Input tensor dtype (float32/float16/bfloat16) / 输入张量数据类型 | ✅ All test files |
| `dim` | Hidden dimension size / 隐藏层维度大小 | ✅ test_causal_conv1d.py, test_mamba_ssm.py |
| `dstate` | State dimension / 状态维度 | ✅ test_mamba_ssm.py |
| `seqlen` | Sequence length / 序列长度 | ✅ All test files |
| `batch_size` | Batch size / 批次大小 | ✅ All test files |
| `has_bias` | Whether to use bias / 是否使用偏置 | ✅ test_causal_conv1d.py |
| `silu_activation` | SILU activation toggle / SILU 激活开关 | ✅ test_causal_conv1d.py |
| `has_z` | Z gate toggle / Z 门控开关 | ✅ test_mamba_ssm.py |
| `ngroups` | Number of groups for head grouping / 头分组数量 | ✅ test_mamba_ssm.py |
| `n_heads` | Number of attention heads / 注意力头数量 | ✅ test_mamba_ssm_ssd.py |
| `d_head` | Head dimension / 头维度 | ✅ test_mamba_ssm_ssd.py |
| `chunk_size` | Chunk size for scan operations / 扫描操作块大小 | ✅ test_mamba_ssm_ssd.py |
| `width` | Convolution kernel width / 卷积核宽度 | ✅ test_causal_conv1d.py |
| `with_padding` | Padding test toggle / 填充测试开关 | ✅ test_causal_conv1d.py, test_mamba_ssm.py |

---

## Test Function Points / 测试功能点

### 1. Causal Conv1D Update Test / 因果卷积1D更新测试 (test_causal_conv1d.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Validate the causal 1D convolution update operation correctness against reference implementation / 验证因果1D卷积更新操作的正确性，与参考实现对比

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.bfloat16
- `silu_activation`: False, True
- `has_bias`: False, True
- `seqlen`: 1
- `width`: 4
- `dim`: 2048, 2048 + 16, 4096

**Function Points / 功能点**:
- Single token convolution update / 单 token 卷积更新
- Bias addition verification / 偏置加法验证
- SILU activation application / SILU 激活应用
- State preservation check / 状态保留检查

**Observable Points / 可观察点**:
- Output tensor correctness (allclose with rtol=3e-3, atol=5e-3) / 输出张量正确性
- Conv state equality with reference / 卷积状态与参考相等
- Activation function output range / 激活函数输出范围

---

### 2. Causal Conv1D Update with Batch Gather Test / 因果卷积1D批量收集更新测试 (test_causal_conv1d.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Test causal conv1d update with batch gathering and padding support / 测试带批量收集和填充支持的因果卷积1D更新

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.float32, torch.float16, torch.bfloat16
- `silu_activation`: False, True
- `has_bias`: False, True
- `seqlen`: 1, 3
- `width`: 3, 4
- `dim`: 2048 + 16, 4096
- `with_padding`: True, False
- `batch_size`: 3

**Function Points / 功能点**:
- Batch index-based state gathering / 基于批次索引的状态收集
- Padding slot handling (PAD_SLOT_ID) / 填充槽位处理
- Multi-sequence convolution / 多序列卷积
- Unused state preservation / 未使用状态保留

**Observable Points / 可观察点**:
- Gathered state correctness / 收集状态正确性
- Padding slot isolation / 填充槽位隔离
- Batch-wise output consistency / 批次输出一致性

---

### 3. Causal Conv1D Variable Length Test / 因果卷积1D变长测试 (test_causal_conv1d.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Validate causal conv1d with variable sequence lengths and cache management / 验证变长序列和缓存管理的因果卷积1D

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.bfloat16
- `silu_activation`: True
- `has_bias`: True
- `width`: 4
- `seqlen`: 8, 30, 249, 2049, 4096
- `dim`: 64, 4096
- `with_padding`: True, False
- `batch`: 4, 10

**Function Points / 功能点**:
- Variable sequence length handling / 变长序列处理
- Cumulative sequence length tracking / 累积序列长度跟踪
- Cache state initialization / 缓存状态初始化
- Initial state flag handling / 初始状态标志处理

**Observable Points / 可观察点**:
- Final states correctness / 最终状态正确性
- Output tensor allclose validation / 输出张量近似相等验证
- Cache index mapping accuracy / 缓存索引映射准确性

---

### 4. Selective State Update Test / 选择性状态更新测试 (test_mamba_ssm.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Verify selective state update (SSU) kernel correctness / 验证选择性状态更新 (SSU) 内核正确性

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.float32, torch.float16, torch.bfloat16
- `has_z`: False, True
- `dstate`: 16, 32, 64
- `dim`: 2048, 2048 + 16, 4096

**Function Points / 功能点**:
- State update with discretization / 离散化状态更新
- dt (delta time) softplus transformation / dt（时间增量）softplus 变换
- A, B, C parameter application / A, B, C 参数应用
- D skip connection / D 跳跃连接
- Z gate multiplication / Z 门控乘法

**Observable Points / 可观察点**:
- State tensor allclose (rtol=5e-3, atol=1e-2 for fp16) / 状态张量近似相等
- Output tensor correctness / 输出张量正确性
- dt_bias application effect / dt_bias 应用效果

---

### 5. Selective State Update with Batch Indices Test / 带批次索引的选择性状态更新测试 (test_mamba_ssm.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Test SSU with batch indices and padding support / 测试带批次索引和填充支持的 SSU

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.float32, torch.float16, torch.bfloat16
- `has_z`: True
- `dstate`: 16, 32, 64
- `dim`: 2048, 2048 + 16, 4096
- `with_padding`: True, False

**Function Points / 功能点**:
- Batch index-based state selection / 基于批次索引的状态选择
- Padding slot isolation (PAD_SLOT_ID) / 填充槽位隔离
- Multi-dimensional state handling / 多维状态处理
- Unused entry preservation / 未使用条目保留

**Observable Points / 可观察点**:
- State diff max/mean logging / 状态差异最大/平均值日志
- Output diff max/mean logging / 输出差异最大/平均值日志
- Padded entries unchanged / 填充条目保持不变

---

### 6. Selective State Update with Heads and Batch Indices Test / 带头和批次索引的选择性状态更新测试 (test_mamba_ssm.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Validate SSU with multi-head attention and batch indices / 验证多头注意力和批次索引的 SSU

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.float32, torch.float16, torch.bfloat16
- `has_z`: False, True
- `tie_hdim`: False, True
- `ngroups`: 1, 2, 4
- `dstate`: 16, 32, 64
- `dim`: 2048, 4096

**Function Points / 功能点**:
- Multi-head state management / 多头状态管理
- Head dimension tying / 头维度绑定
- Group-based B/C broadcasting / 基于组的 B/C 广播
- Per-head dt, A, D parameters / 每头 dt, A, D 参数

**Observable Points / 可观察点**:
- Output max/mean diff logging / 输出最大/平均差异日志
- Multi-head state consistency / 多头状态一致性
- Group broadcasting correctness / 组广播正确性

---

### 7. Mamba Chunk Scan Single Example Test / Mamba 块扫描单例测试 (test_mamba_ssm_ssd.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Test mamba chunk scan kernel on single examples without batching / 在单例上测试 mamba 块扫描内核（无批处理）

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.float32, torch.float16, torch.bfloat16 (CI: torch.float32, torch.bfloat16)
- `n_heads`: 3, 4, 11, 16, 32 (CI: 3, 32)
- `d_head`: 5, 8, 19, 32, 128 (CI: 5, 128)
- `seq_len_chunk_size`: (112, 16), (128, 32) (CI: (112, 16))

**Function Points / 功能点**:
- Chunked scan computation / 分块扫描计算
- SSD (State Space Discretization) minimal reference comparison / SSD 最小参考对比
- Final state extraction / 最终状态提取
- Sequence output validation / 序列输出验证

**Observable Points / 可观察点**:
- Y output last token match / Y 输出最后 token 匹配
- Final state dtype conversion (fp32) / 最终状态数据类型转换
- bfloat16 tolerance handling (atol=5e-2) / bfloat16 容差处理

---

### 8. Mamba Chunk Scan Continuous Batch Test / Mamba 块扫描连续批次测试 (test_mamba_ssm_ssd.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**: Test mamba chunk scan with continuous batching (chunked prefill) / 测试连续批处理的 mamba 块扫描（分块预填充）

**Test Type / 测试类型**: Integration Test / 集成测试 🔗

**Covered Parameters / 覆盖参数**:
- `itype`: torch.float32, torch.float16 (CI: torch.float32)
- `n_heads`: 4, 8, 13 (CI: 4, 13)
- `d_head`: 5, 16, 21, 32 (CI: 5, 32)
- `seq_len_chunk_size_cases`: Various (seqlen, chunk_size, num_examples, cases)

**Function Points / 功能点**:
- Continuous batch generation / 连续批次生成
- Example exhaustion tracking / 示例耗尽跟踪
- State persistence across chunks / 跨块状态持久化
- cu_seqlens and seq_idx handling / cu_seqlens 和 seq_idx 处理
- Chunk indices and offsets computation / 块索引和偏移计算

**Observable Points / 可观察点**:
- Per-example output validation / 每示例输出验证
- State reset on exhaustion / 耗尽时状态重置
- Cross-chunk consistency / 跨块一致性

---

### 9. Mamba Chunk Scan Continuous Batch Prefill Chunking Test / Mamba 块扫描连续批次预填充分块测试 (test_mamba_ssm_ssd.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**: Verify chunked prefill correctness by comparing chunked vs full sequence results / 通过对比分块与完整序列结果验证分块预填充正确性

**Test Type / 测试类型**: Integration Test / 集成测试 🔗

**Covered Parameters / 覆盖参数**:
- `chunk_size`: 8, 256
- `seqlens`: (16, 2, 8, 13), (270, 88, 212, 203), (16, 20)

**Function Points / 功能点**:
- Sequence splitting and reconstruction / 序列分割与重建
- Partial state carry-over / 部分状态传递
- Concatenation validation / 拼接验证
- Two-phase computation (partial + remaining) / 两阶段计算（部分 + 剩余）

**Observable Points / 可观察点**:
- Chunked vs full output equality / 分块与完整输出相等性
- State consistency across phases / 跨阶段状态一致性
- Sequence boundary handling / 序列边界处理

---

### 10. Mamba Chunk Scan Intermediate States Test / Mamba 块扫描中间状态测试 (test_mamba_ssm_ssd.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**: Validate intermediate states extraction during chunk scan / 验证块扫描期间的中间状态提取

**Test Type / 测试类型**: Unit Test / 单元测试 🔬

**Covered Parameters / 覆盖参数**:
- `itype`: torch.float32, torch.bfloat16
- `n_heads`: 4, 16
- `d_head`: 32, 64
- `seq_len_chunk_size`: (128, 32), (256, 64)

**Function Points / 功能点**:
- Intermediate states return / 中间状态返回
- Per-chunk state validation / 每块状态验证
- Final state extraction / 最终状态提取
- States shape verification (batch, num_chunks, n_heads, d_head, d_head) / 状态形状验证

**Observable Points / 可观察点**:
- Each chunk state correctness / 每块状态正确性
- Final state dtype (fp32) / 最终状态数据类型
- States tensor dimensions / 状态张量维度

---

### 11. Mixer2 Gated Norm Multi-GPU Test / Mixer2 门控归一化多GPU测试 (test_mamba2_mixer.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**: Test Mixer2 RMSNorm gated operation with tensor parallelism across multiple GPUs / 测试跨多GPU的张量并行 Mixer2 RMSNorm 门控操作

**Test Type / 测试类型**: Integration Test / 集成测试 🔗

**Covered Parameters / 覆盖参数**:
- `batch_size`: 8
- `seq_len`: 128
- `hidden_size_n_groups`: (64, 1), (100, 4)
- `dtype`: torch.float16
- `NUM_GPUS`: 2

**Function Points / 功能点**:
- Multi-process distributed setup / 多进程分布式设置
- Tensor parallelism (TP) weight loading / 张量并行权重加载
- Gated normalization computation / 门控归一化计算
- TP vs single-GPU reference comparison / TP 与单GPU参考对比

**Observable Points / 可观察点**:
- Output close validation (atol=5e-3, rtol=1e-3) / 输出近似验证
- TP rank-specific output slice / TP 等级特定输出切片
- Distributed environment initialization / 分布式环境初始化

---

## Observable Points Summary / 可观察点汇总

### Numerical Correctness Observables / 数值正确性可观察点

| Observable / 可观察点 | Description / 描述 | Test Files / 测试文件 |
|---------------------|-------------------|---------------------|
| Output tensor allclose / 输出张量近似相等 | Compare kernel output with reference / 对比内核输出与参考 | All |
| State tensor equality / 状态张量相等 | Verify state preservation / 验证状态保留 | test_mamba_ssm.py, test_causal_conv1d.py |
| Final state correctness / 最终状态正确性 | Validate final SSM state / 验证最终 SSM 状态 | test_mamba_ssm_ssd.py |
| Diff max/mean logging / 差异最大/平均日志 | Track numerical deviations / 跟踪数值偏差 | test_mamba_ssm.py |

### Memory and State Observables / 内存和状态可观察点

| Observable / 可观察点 | Description / 描述 | Test Files / 测试文件 |
|---------------------|-------------------|---------------------|
| Padding isolation / 填充隔离 | PAD_SLOT_ID entries unchanged / PAD_SLOT_ID 条目不变 | test_causal_conv1d.py, test_mamba_ssm.py |
| Cache state management / 缓存状态管理 | Conv state gathering/scattering / 卷积状态收集/分散 | test_causal_conv1d.py |
| State persistence / 状态持久化 | Cross-chunk state carry-over / 跨块状态传递 | test_mamba_ssm_ssd.py |
| Unused entry preservation / 未使用条目保留 | Non-indexed entries unchanged / 非索引条目不变 | test_causal_conv1d.py, test_mamba_ssm.py |

### Performance and Compatibility Observables / 性能和兼容性可观察点

| Observable / 可观察点 | Description / 描述 | Test Files / 测试文件 |
|---------------------|-------------------|---------------------|
| Device compatibility / 设备兼容性 | CUDA/XPU/AMD support / CUDA/XPU/AMD 支持 | All |
| Dtype tolerance / 数据类型容差 | fp32/fp16/bf16 thresholds / fp32/fp16/bf16 阈值 | All |
| CI parameter reduction / CI 参数缩减 | Reduced test matrix in CI / CI 中缩减测试矩阵 | test_mamba_ssm_ssd.py |
| Multi-GPU synchronization / 多GPU同步 | TP correctness across ranks / 跨等级 TP 正确性 | test_mamba2_mixer.py |

---

## Test File Summary / 测试文件汇总

| # | Test File / 测试文件 | Main Function / 主函数 | Test Type / 测试类型 | Category / 类别 |
|---|---------------------|----------------------|---------------------|----------------|
| 1 | test_causal_conv1d.py | test_causal_conv1d_update | Unit Test | Causal Conv |
| 2 | test_causal_conv1d.py | test_causal_conv1d_update_with_batch_gather | Unit Test | Causal Conv |
| 3 | test_causal_conv1d.py | test_causal_conv1d_varlen | Unit Test | Causal Conv |
| 4 | test_mamba_ssm.py | test_selective_state_update | Unit Test | SSU |
| 5 | test_mamba_ssm.py | test_selective_state_update_with_batch_indices | Unit Test | SSU |
| 6 | test_mamba_ssm.py | test_selective_state_update_with_heads_with_batch_indices | Unit Test | SSU |
| 7 | test_mamba_ssm_ssd.py | test_mamba_chunk_scan_single_example | Unit Test | SSD |
| 8 | test_mamba_ssm_ssd.py | test_mamba_chunk_scan_cont_batch | Integration Test | SSD |
| 9 | test_mamba_ssm_ssd.py | test_mamba_chunk_scan_cont_batch_prefill_chunking | Integration Test | SSD |
| 10 | test_mamba_ssm_ssd.py | test_mamba_chunk_scan_intermediate_states | Unit Test | SSD |
| 11 | test_mamba2_mixer.py | test_mixer2_gated_norm_multi_gpu | Integration Test | Mixer2 |

---

## Test Infrastructure / 测试基础设施

### CI Registration / CI 注册
All tests are registered with CI estimations for both CUDA and AMD platforms:
- `register_cuda_ci(est_time, suite)` - CUDA CI registration
- `register_amd_ci(est_time, suite)` - AMD CI registration

### Conftest.py Fixtures / Conftest.py 固件
- `_init_mamba_ssu_backend`: Session-scoped fixture to initialize Mamba SSU dispatch backend / 会话级固件，用于初始化 Mamba SSU 调度后端

### Reference Implementations / 参考实现
- `causal_conv1d_ref`: PyTorch reference for causal convolution / 因果卷积的 PyTorch 参考
- `causal_conv1d_update_ref`: PyTorch reference for conv update / 卷积更新的 PyTorch 参考
- `selective_state_update_ref`: PyTorch reference for SSU / SSU 的 PyTorch 参考
- `ssd_minimal_discrete`: Minimal torch implementation for SSD / SSD 的最小 torch 实现

---

## Notes / 备注

1. **Device Requirements / 设备要求**: Most tests require CUDA or XPU devices and skip on other platforms / 大多数测试需要 CUDA 或 XPU 设备，在其他平台上跳过

2. **Tolerance Variations / 容差变化**: Different tolerances are used based on dtype:
   - float32: rtol=3e-4, atol=1e-3
   - float16: rtol=5e-3, atol=1e-2
   - bfloat16: rtol=1e-2, atol=5e-2 (or higher for some tests)

3. **CI Optimization / CI 优化**: Test parameters are reduced in CI environment (checked via `is_in_ci()`) to minimize test time / 在 CI 环境中减少测试参数以最小化测试时间

4. **AMD Specifics / AMD 特性**: Some tests have AMD-specific adjustments (e.g., `AMDGCN_USE_BUFFER_OPS` environment variable) / 某些测试具有 AMD 特定调整
