# Layers Testing Knowledge Base / 层测试知识库

## Overview / 概述

This directory contains test files for various neural network layer implementations in SGLang, focusing on advanced attention mechanisms including Mamba/State Space Models (SSM) and layer normalization operations.

本目录包含 SGLang 中各种神经网络层实现的测试文件，专注于高级注意力机制，包括 Mamba/状态空间模型（SSM）和层归一化操作。

### Test Categories / 测试类别

- **Mamba/SSM Layers**: Causal 1D convolution, selective state update, Mamba chunk scan (SSD), Mamba2 mixer
- **Normalization Layers**: LayerNorm and RMSNorm with gating support
- **Multi-GPU Tests**: Tensor parallelism tests for distributed inference

- **Mamba/SSM 层**: 因果 1D 卷积、选择性状态更新、Mamba 块扫描（SSD）、Mamba2 混合器
- **归一化层**: 支持门控的 LayerNorm 和 RMSNorm
- **多 GPU 测试**: 分布式推理的张量并行测试

### Supported Architectures / 支持的架构

- CUDA (NVIDIA GPUs)
- AMD GPUs (ROCm/HIP)
- XPU (Intel)

---

## Core Parameters / 核心参数

| Parameter / 参数 | Description / 描述 | Test Coverage / 测试覆盖 |
|------------------|-------------------|-------------------------|
| `itype` (dtype) | Input data type (float32, float16, bfloat16) / 输入数据类型 | ✅ test_causal_conv1d.py, test_mamba_ssm.py, test_mamba_ssm_ssd.py |
| `seqlen` | Sequence length / 序列长度 | ✅ test_causal_conv1d.py, test_mamba_ssm_ssd.py |
| `dim` | Hidden dimension size / 隐藏维度大小 | ✅ test_causal_conv1d.py, test_mamba_ssm.py |
| `batch_size` | Batch size for inference / 推理批次大小 | ✅ All test files |
| `width` | Convolution kernel width / 卷积核宽度 | ✅ test_causal_conv1d.py |
| `dstate` | SSM state dimension / SSM 状态维度 | ✅ test_mamba_ssm.py |
| `chunk_size` | Chunk size for SSD computation / SSD 计算块大小 | ✅ test_mamba_ssm_ssd.py |
| `hidden_size` | Hidden layer size / 隐藏层大小 | ✅ test_fla_layernorm_guard.py |
| `group_size` | Group size for group normalization / 组归一化的组大小 | ✅ test_fla_layernorm_guard.py |
| `activation` | Activation function (silu, None) / 激活函数 | ✅ test_causal_conv1d.py |

---

## Test Function Points / 测试功能点

### 1. Causal 1D Convolution Update / 因果 1D 卷积更新 (test_causal_conv1d.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**: 
Verifies the correctness of the causal 1D convolution update operation by comparing with PyTorch reference implementation.

验证因果 1D 卷积更新操作的正确性，通过与 PyTorch 参考实现进行比较。

**Covered Parameters / 覆盖参数**:
- `dim`: [2048, 2048 + 16, 4096]
- `width`: [4]
- `seqlen`: [1]
- `has_bias`: [False, True]
- `silu_activation`: [False, True]
- `itype`: [torch.bfloat16]

**Function Points / 功能点**:
- Causal convolution with Silu activation / 带有 Silu 激活的因果卷积
- Bias addition support / 偏置加法支持
- State management for incremental decoding / 增量解码的状态管理
- Numerical precision validation (bfloat16) / 数值精度验证 (bfloat16)

**Observable Points / 可观察点**:
- State tensor equality between kernel and reference / 内核与参考实现之间的状态张量相等性
- Output tensor allclose within tolerance / 容差范围内的输出张量近似性
- Memory corruption check (unused states unchanged) / 内存损坏检查（未使用状态未改变）

---

### 2. Causal 1D Convolution with Batch Gather / 因果 1D 卷积与批次收集 (test_causal_conv1d.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**:
Tests causal convolution with batch gather indices for variable-length sequence handling in continuous batching.

测试带有批次收集索引的因果卷积，用于连续批处理中的变长序列处理。

**Covered Parameters / 覆盖参数**:
- `batch_size`: [3]
- `with_padding`: [True, False]
- `dim`: [2048 + 16, 4096]
- `width`: [3, 4]
- `seqlen`: [1, 3]
- `itype`: [torch.float32, torch.float16, torch.bfloat16]

**Function Points / 功能点**:
- Continuous batching support with padding mask / 带有填充掩码的连续批处理支持
- Cache slot indexing via `conv_state_indices` / 通过 `conv_state_indices` 进行缓存槽索引
- Padding slot handling (PAD_SLOT_ID) / 填充槽处理 (PAD_SLOT_ID)
- Reference implementation validation / 参考实现验证

**Observable Points / 可观察点**:
- Padded slots are ignored in computation / 填充槽在计算中被忽略
- Used slots match reference output / 使用的槽与参考输出匹配
- Unused slots remain unchanged / 未使用的槽保持不变

---

### 3. Variable Length Causal 1D Convolution / 变长因果 1D 卷积 (test_causal_conv1d.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**:
Tests causal 1D convolution with variable-length sequences using cumulative sequence lengths.

测试使用累积序列长度的变长序列的因果 1D 卷积。

**Covered Parameters / 覆盖参数**:
- `batch`: [4, 10]
- `seqlen`: [8, 30, 249, 2049, 4096]
- `dim`: [64, 4096]
- `width`: [4]

**Function Points / 功能点**:
- Variable length sequence handling via `query_start_loc` / 通过 `query_start_loc` 处理变长序列
- Initial state support per sequence / 每个序列的初始状态支持
- Final state output for KV cache / KV 缓存的最终状态输出
- State indexing for cache management / 缓存管理的状态索引

**Observable Points / 可观察点**:
- Correct variable-length output shapes / 正确的变长输出形状
- Final states stored in correct cache slots / 最终状态存储在正确的缓存槽中
- Numerical accuracy for long sequences / 长序列的数值精度

---

### 4. Mamba2 Mixer Gated Norm Multi-GPU / Mamba2 混合器门控归一化多 GPU (test_mamba2_mixer.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**:
Tests tensor parallelism for the Mixer2 gated normalization layer across multiple GPUs.

测试跨多 GPU 的 Mixer2 门控归一化层的张量并行。

**Covered Parameters / 覆盖参数**:
- `batch_size`: [8]
- `seq_len`: [128]
- `hidden_size_n_groups`: [(64, 1), (100, 4)]
- `dtype`: [torch.float16]
- `NUM_GPUS`: 2

**Function Points / 功能点**:
- Distributed tensor parallelism initialization / 分布式张量并行初始化
- Multi-GPU parallel environment setup / 多 GPU 并行环境设置
- Gated normalization with Split/Column parallelism / 带有拆分/列并行化的门控归一化
- All-reduce communication verification / 全局归约通信验证

**Observable Points / 可观察点**:
- Multi-GPU output matches single-GPU reference / 多 GPU 输出与单 GPU 参考匹配
- Correct tensor sharding across devices / 跨设备的正确张量分片
- Communication pattern validation / 通信模式验证

---

### 5. Selective State Update / 选择性状态更新 (test_mamba_ssm.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**:
Verifies the selective state update SSM kernel by comparing with PyTorch reference implementation.

通过与 PyTorch 参考实现进行比较，验证选择性状态更新 SSM 内核。

**Covered Parameters / 覆盖参数**:
- `dim`: [2048, 2048 + 16, 4096]
- `dstate`: [16, 32, 64]
- `has_z`: [False, True]
- `itype`: [torch.float32, torch.float16, torch.bfloat16]

**Function Points / 功能点**:
- State discretization via dt (time step) / 通过 dt（时间步）进行状态离散化
- Parameter matrices: A, B, C, D / 参数矩阵：A、B、C、D
- Softplus activation on dt / dt 上的 Softplus 激活
- Z-gate multiplication (when present) / Z 门乘（如果存在）

**Observable Points / 可观察点**:
- State tensor updated correctly / 状态张量正确更新
- Output tensor matches reference / 输出张量与参考匹配
- Numerical stability across precisions / 跨精度的数值稳定性

---

### 6. Selective State Update with Batch Indices / 带批次索引的选择性状态更新 (test_mamba_ssm.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**:
Tests selective state update with batch indices for continuous batching and cache management.

测试带有批次索引的选择性状态更新，用于连续批处理和缓存管理。

**Covered Parameters / 覆盖参数**:
- `batch_size`: 3
- `with_padding`: [True, False]
- `itype`: [torch.float32, torch.float16, torch.bfloat16]

**Function Points / 功能点**:
- State batch indexing for cache slots / 缓存槽的状态批次索引
- Padding mask support for batched processing / 批处理的填充掩码支持
- PAD_SLOT_ID handling for unused entries / 未使用条目的 PAD_SLOT_ID 处理

**Observable Points / 可观察点**:
- Indexed states updated correctly / 索引状态正确更新
- Padding entries unchanged / 填充条目未改变
- Numerical accuracy with batching / 批处理的数值精度

---

### 7. Selective State Update with Heads / 带头部的选择性状态更新 (test_mamba_ssm.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**:
Tests selective state update with multi-head attention support (group query attention).

测试带有头部维度的选择性状态更新（组查询注意力）。

**Covered Parameters / 覆盖参数**:
- `dim`: [2048, 4096]
- `dstate`: [16, 32, 64]
- `ngroups`: [1, 2, 4]
- `tie_hdim`: [False, True]
- `has_z`: [False, True]
- `itype`: [torch.float32, torch.float16, torch.bfloat16]

**Function Points / 功能点**:
- Multi-head state dimension (4D tensors) / 多头状态维度（4D 张量）
- Group query attention support / 组查询注意力支持
- Head dimension tying (headdim parameter) / 头维度绑定 (headdim 参数)

**Observable Points / 可观察点**:
- Correct 4D tensor shape handling / 正确的 4D 张量形状处理
- Head-wise state update accuracy / 逐头状态更新精度
- Group-wise parameter broadcast / 逐组参数广播

---

### 8. Mamba Chunk Scan Single Example / Mamba 块扫描单示例 (test_mamba_ssm_ssd.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**:
Tests the mamba_chunk_scan_combined kernel on single examples against a minimal reference implementation.

在单示例上测试 mamba_chunk_scan_combined 内核与最小参考实现。

**Covered Parameters / 覆盖参数**:
- `itype`: [torch.float32, torch.float16, bfloat16] (CI: [torch.float32, torch.bfloat16])
- `n_heads`: [3, 4, 11, 16, 32] (CI: [3, 32])
- `d_head`: [5, 8, 19, 32, 128] (CI: [5, 128])
- `seq_len_chunk_size`: [(112, 16), (128, 32)] (CI: [(112, 16)])

**Function Points / 功能点**:
- SSD (State Space Decomposition) scan / SSD（状态空间分解）扫描
- Block-wise chunk decomposition / 块级块分解
- Segment sum computation for state transitions / 状态转换的段和计算
- Final state output for KV caching / KV 缓存的最终状态输出

**Observable Points / 可观察点**:
- Output matches reference at sequence positions / 序列位置的输出与参考匹配
- Final state numerical accuracy / 最终状态数值精度
- FP32 state casting in kernel / 内核中的 FP32 状态转换

---

### 9. Mamba Chunk Scan Continuous Batch / Mamba 块扫描连续批处理 (test_mamba_ssm_ssd.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**:
Tests mamba chunk scan with continuous batching for chunked prefill scenarios.

测试带有连续批处理的 Mamba 块扫描，用于分块预填充场景。

**Covered Parameters / 覆盖参数**:
- `itype`: [torch.float32, torch.float16]
- `n_heads`: [4, 8, 13]
- `d_head`: [5, 16, 21, 32]
- Various `seq_len_chunk_size_cases` combinations / 各种 `seq_len_chunk_size_cases` 组合

**Function Points / 功能点**:
- Continuous batching with example cycling / 带有示例循环的连续批处理
- Initial states from previous chunks / 来自先前块的初始状态
- State reset for exhausted examples / 耗尽示例的状态重置
- Cumulative sequence length metadata / 累积序列长度元数据

**Observable Points / 可观察点**:
- State persistence across chunks / 跨块的状态持久性
- Correct state reset for cycling examples / 循环示例的正确状态重置
- Numerical accuracy with multi-example batches / 多示例批次的数值精度

---

### 10. Mamba Chunk Scan Prefill Chunking / Mamba 块扫描预填充分块 (test_mamba_ssm_ssd.py) [Integration Test / 集成测试]

**Test Goal / 测试目标**:
Verifies chunked prefill correctness by comparing concatenation of chunked results with full sequence results.

通过比较块结果的连接与完整序列结果，验证分块预填充的正确性。

**Covered Parameters / 覆盖参数**:
- `chunk_size`: [8, 256]
- `seqlens`: [(16, 2, 8, 13), (270, 88, 212, 203), (16, 20)]

**Function Points / 功能点**:
- Sequence chunking at arbitrary positions / 任意位置的序列分块
- Initial state passing between chunks / 块之间的初始状态传递
- Variable sequence length handling / 可变序列长度处理
- cu_seqlens and seq_idx metadata computation / cu_seqlens 和 seq_idx 元数据计算

**Observable Points / 可观察点**:
- Full sequence output equals concatenated chunked output / 完整序列输出等于连接的块输出
- State consistency across chunk boundaries / 跨块边界的状态一致性
- Correct chunk_indices/chunk_offsets computation / 正确的 chunk_indices/chunk_offsets 计算

---

### 11. LayerNorm Guard Forward / 层归一化前向 (test_fla_layernorm_guard.py) [Precision Test / 精度测试]

**Test Goal / 测试目标**:
Tests FLA (Flash Linear Attention) LayerNorm and RMSNorm with gating support for multi-GPU tensor parallelism.

测试带有门控支持的 FLA（闪存线性注意力）LayerNorm 和 RMSNorm，用于多 GPU 张量并行。

**Covered Parameters / 覆盖参数**:
- `num_tokens`: [128, 513]
- `hidden_size`: [256, 1024]
- `dtype`: [torch.bfloat16]
- `case`: Various FwdCase combinations (LayerNorm, RMSNorm, gated variants, group norm)

**Function Points / 功能点**:
- LayerNorm and RMSNorm computation / LayerNorm 和 RMSNorm 计算
- Gating support (pre-mul and post-mul SiLU) / 门控支持（前乘和后乘 SiLU）
- Group normalization (group_size parameter) / 组归一化 (group_size 参数)
- Multi-GPU tensor parallelism / 多 GPU 张量并行

**Observable Points / 可观察点**:
- Output matches reference implementation / 输出与参考实现匹配
- mean/rstd shape correctness (group vs full) / mean/rstd 形状正确性（组与完整）
- Row-wise computation for arbitrary token counts / 任意标记数的逐行计算
- Strided input handling / 步幅输入处理

---

### 12. LayerNorm Guard Miscellaneous / 层归一化其他测试 (test_fla_layernorm_guard.py) [Unit Test / 单元测试]

**Test Goal / 测试目标**:
Tests additional edge cases for LayerNorm including output buffer reuse and multidimensional inputs.

测试 LayerNorm 的其他边缘情况，包括输出缓冲区重用和多维输入。

**Covered Parameters / 覆盖参数**:
- Row-wise block sizes: 513 / 逐行块大小：513
- Strided input / 步幅输入
- Output buffer provision / 输出缓冲区提供
- Multidimensional shapes: [(4, 16, 1024)]

**Function Points / 功能点**:
- Non-power-of-2 token counts / 非 2 幂次方的标记数
- Strided tensor handling / 步幅张量处理
- External output buffer support / 外部输出缓冲区支持
- Autograd function interface / 自动求导函数接口

**Observable Points / 可观察点**:
- Output buffer memory sharing / 输出缓冲区内存共享
- Shape preservation through autograd / 通过自动求导保持形状
- Numerical accuracy for edge cases / 边缘情况的数值精度

---

## Observable Points Summary / 可观察点汇总

### Server-side Observables / 服务端可观察点
| Observable / 可观察点 | Description / 描述 | Relevant Tests / 相关测试 |
|---------------------|-------------------|-------------------------|
| State cache slots | Accuracy of cache slot indexing for KV states | test_mamba_ssm.py, test_causal_conv1d.py |
| Tensor parallelism | Correct device sharding and communication | test_mamba2_mixer.py, test_fla_layernorm_guard.py |
| Memory corruption | Unused cache slots remain unchanged | test_mamba_ssm.py, test_causal_conv1d.py |

### Inference Observables / 推理可观察点
| Observable / 可观察点 | Description / 描述 | Relevant Tests / 相关测试 |
|---------------------|-------------------|-------------------------|
| Output correctness | Final output matches reference | All tests |
| Numerical precision | Tolerance adherence for different dtypes | test_mamba_ssm.py, test_causal_conv1d.py |
| State propagation | Internal state updates correctly | test_mamba_ssm.py, test_mamba_ssm_ssd.py |
| Chunk consistency | Chunked results equal full sequence | test_mamba_ssm_ssd.py |

### Performance Observables / 性能可观察点
| Observable / 可观察点 | Description / 描述 | Relevant Tests / 相关测试 |
|---------------------|-------------------|-------------------------|
| Sequence length scaling | Accuracy across short/long sequences | test_causal_conv1d.py, test_mamba_ssm_ssd.py |
| Batch processing | Efficiency with padding and batching | test_mamba_ssm.py, test_mamba_ssm_ssd.py |
| Memory reuse | Output buffer sharing efficiency | test_fla_layernorm_guard.py |

### Error Observables / 错误可观察点
| Observable / 可观察点 | Description / 描述 | Relevant Tests / 相关测试 |
|---------------------|-------------------|-------------------------|
| Tolerance violations | Assertion failures for allclose checks | All precision tests |
| Shape mismatches | Incorrect tensor dimensions | All tests |
| Index out of bounds | Invalid batch/slot indices | test_mamba_ssm.py, test_causal_conv1d.py |

---

## Test File Summary / 测试文件汇总

| # | Test File / 测试文件 | Main Functions / 主函数 | Test Type / 测试类型 | Category / 类别 |
|---|---------------------|------------------------|---------------------|----------------|
| 1 | [conftest.py](mamba/conftest.py) | `_init_mamba_ssu_backend` | Setup / 设置 | Mamba |
| 2 | [test_causal_conv1d.py](mamba/test_causal_conv1d.py) | `test_causal_conv1d_update`, `test_causal_conv1d_update_with_batch_gather`, `test_causal_conv1d_varlen` | Precision / 精度测试 | Mamba |
| 3 | [test_mamba2_mixer.py](mamba/test_mamba2_mixer.py) | `test_mixer2_gated_norm_multi_gpu` | Integration / 集成测试 | Mamba2 |
| 4 | [test_mamba_ssm.py](mamba/test_mamba_ssm.py) | `test_selective_state_update`, `test_selective_state_update_with_batch_indices`, `test_selective_state_update_with_heads_with_batch_indices` | Precision / 精度测试 | Mamba SSM |
| 5 | [test_mamba_ssm_ssd.py](mamba/test_mamba_ssm_ssd.py) | `test_mamba_chunk_scan_single_example`, `test_mamba_chunk_scan_cont_batch`, `test_mamba_chunk_scan_cont_batch_prefill_chunking` | Integration / 集成测试 | Mamba SSD |
| 6 | [test_fla_layernorm_guard.py](test_fla_layernorm_guard.py) | `test_layernorm_guard_fwd_spawn`, `test_layernorm_guard_misc_spawn` | Precision / 精度测试 | Normalization |

---

## Key Implementations Referenced / 引用的关键实现

### Internal Modules / 内部模块

| Module Path / 模块路径 | Purpose / 用途 |
|---------------------|---------------|
| `sglang.srt.layers.attention.mamba.causal_conv1d_triton` | Causal 1D convolution kernels / 因果 1D 卷积内核 |
| `sglang.srt.layers.attention.mamba.ops` | SSM operations (selective_state_update, mamba_chunk_scan_combined) / SSM 操作 |
| `sglang.srt.layers.attention.mamba.mixer2_rms_norm_gated` | Mamba2 gated normalization / Mamba2 门控归一化 |
| `sglang.srt.layers.attention.mamba.mamba2_metadata` | Mamba2 metadata computation / Mamba2 元数据计算 |
| `sglang.srt.layers.attention.fla.layernorm_gated` | FLA LayerNorm/RMSNorm with gating / FLA 带门控的 LayerNorm/RMSNorm |
| `sglang.srt.distributed.parallel_state` | Distributed parallel state management / 分布式并行状态管理 |

### External References / 外部参考

- `vllm-project/vllm` - Mamba kernel test patterns (MIT License)
- `state-spaces/mamba` - SSD minimal implementation reference

---

## CI Registration / CI 注册

Files register for CI using:

| Test File | CUDA Suite | AMD Suite | Estimated Time |
|-----------|------------|-----------|----------------|
| test_causal_conv1d.py | stage-b-test-1-gpu-small | stage-b-test-1-gpu-small-amd | 10s / 20s |
| test_mamba2_mixer.py | stage-b-test-2-gpu-large | - | 32s |
| test_mamba_ssm.py | stage-b-test-1-gpu-small | stage-b-test-1-gpu-small-amd | 10s / 20s |
| test_mamba_ssm_ssd.py | stage-b-test-1-gpu-small | stage-b-test-1-gpu-small-amd | 10s / 34s |
| test_fla_layernorm_guard.py | stage-b-test-2-gpu-large (disabled) | - | 60s (disabled) |

---

*Generated by test-case-analyzer-v3 / 由 test-case-analyzer-v3 生成*
*Date: 2026-05-14 / 日期: 2026-05-14*
