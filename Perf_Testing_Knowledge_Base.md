# Performance Testing Knowledge Base / 性能测试知识库

## Overview / 概述

This knowledge base documents the performance test suite for SGLang, covering various benchmarking scenarios including offline throughput, online latency, VLM performance, API benchmarks, and multi-GPU configurations.

本知识库记录了 SGLang 的性能测试套件，涵盖各种基准测试场景，包括离线吞吐量、在线延迟、VLM 性能、API 基准测试和多 GPU 配置。

### Supported Test Categories / 支持的测试类别

| Category / 类别 | Description / 描述 |
|-----------------|-------------------|
| Offline Throughput / 离线吞吐量 | Batch inference performance / 批处理推理性能 |
| Online Latency / 在线延迟 | Real-time serving latency / 实时服务延迟 |
| VLM Performance / VLM 性能 | Vision-Language model benchmarks / 视觉语言模型基准测试 |
| API Benchmarks / API 基准测试 | Score and Embeddings API performance / Score 和 Embeddings API 性能 |
| Multi-GPU / 多 GPU | Tensor Parallel and Pipeline Parallel tests / 张量并行和流水线并行测试 |
| Speculative Decoding / 推测解码 | EAGLE speculative decoding performance / EAGLE 推测解码性能 |
| LoRA / LoRA | Low-Rank Adaptation serving tests / 低秩适配服务测试 |

---

## Core Parameters / 核心参数

| Parameter / 参数 | Description / 描述 | Test Coverage / 测试覆盖 |
|------------------|-------------------|-------------------------|
| `--tp` | Tensor Parallel size / 张量并行大小 | ✅ test_bench_one_batch_2gpu.py, test_bench_serving_2gpu.py |
| `--pp-size` | Pipeline Parallel size / 流水线并行大小 | ✅ test_bench_serving_2gpu.py |
| `--cuda-graph-max-bs` | CUDA graph max batch size / CUDA 图最大批大小 | ✅ test_bench_one_batch_2gpu.py |
| `--enable-torch-compile` | Enable torch compile / 启用 torch 编译 | ✅ test_bench_one_batch_2gpu.py |
| `--disable-radix-cache` | Disable radix cache / 禁用基数缓存 | ✅ test_bench_serving_1gpu_part1.py, test_bench_serving_2gpu.py |
| `--chunked-prefill-size` | Chunked prefill size / 分块预填充大小 | ✅ test_bench_serving_1gpu_part1.py |
| `--attention-backend` | Attention backend (triton/flashinfer) / 注意力后端 | ✅ test_bench_serving_1gpu_part1.py |
| `--max-running-requests` | Max concurrent requests / 最大并发请求数 | ✅ test_bench_serving_1gpu_part1.py |
| `--enable-lora` | Enable LoRA serving / 启用 LoRA 服务 | ✅ test_bench_serving_1gpu_part1.py |
| `--max-loras-per-batch` | Max LoRAs per batch / 每批最大 LoRA 数 | ✅ test_bench_serving_1gpu_part1.py |
| `--mem-fraction-static` | Static memory fraction / 静态内存比例 | ✅ Multiple tests / 多个测试 |
| `--speculative-algorithm` | Speculative algorithm (EAGLE) / 推测算法 | ✅ test_bench_serving_1gpu_large.py, test_dpsk_v3_fp4_4gpu_perf.py |
| `--speculative-draft-model-path` | Draft model path for speculative / 推测草稿模型路径 | ✅ test_bench_serving_1gpu_large.py |
| `--speculative-num-steps` | Number of speculative steps / 推测步数 | ✅ test_bench_serving_1gpu_large.py, test_dpsk_v3_fp4_4gpu_perf.py |
| `--speculative-eagle-topk` | EAGLE top-k value / EAGLE top-k 值 | ✅ test_bench_serving_1gpu_large.py, test_dpsk_v3_fp4_4gpu_perf.py |
| `--speculative-num-draft-tokens` | Number of draft tokens / 草稿令牌数 | ✅ test_bench_serving_1gpu_large.py, test_dpsk_v3_fp4_4gpu_perf.py |
| `--quantization` | Quantization mode (fp8/fp4) / 量化模式 | ✅ test_bench_serving_2gpu.py, test_dpsk_v3_fp4_4gpu_perf.py |
| `--trust-remote-code` | Trust remote code / 信任远程代码 | ✅ test_dpsk_v3_fp4_4gpu_perf.py, test_vlms_perf.py |

---

## Test Function Points / 测试功能点

### 1. Bench One Batch 2GPU / 单批次 2GPU 基准测试 (test_bench_one_batch_2gpu.py) 📊

**Test Goal / 测试目标**: Validate offline throughput performance for MoE models with Tensor Parallel=2 on dual GPU setup / 验证双 GPU 设置下张量并行=2 的 MoE 模型离线吞吐量性能

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--tp 2`: Tensor parallel size 2 / 张量并行大小为 2
- `--cuda-graph-max-bs 2`: CUDA graph max batch size / CUDA 图最大批大小
- `--enable-torch-compile`: Torch compilation optimization / Torch 编译优化

**Function Points / 功能点**:
- MoE model inference with TP=2 / TP=2 的 MoE 模型推理
- Torch compile optimization for throughput / 用于吞吐量的 Torch 编译优化
- Batch size 1 inference throughput / 批大小为 1 的推理吞吐量

**Observable Points / 可观察点**:
- Output throughput (token/s) / 输出吞吐量 (token/s)
- AMD MI300X threshold: > 85-200 token/s / AMD MI300X 阈值: > 85-200 token/s
- NVIDIA threshold: > 125-220 token/s / NVIDIA 阈值: > 125-220 token/s

---

### 2. Bench Serving 1GPU Large / 单 GPU 大型服务基准测试 (test_bench_serving_1gpu_large.py) 📊

**Test Goal / 测试目标**: Performance tests for single GPU requiring H200 (80GB) - FP8 and EAGLE speculative decoding tests / 需要 H200 (80GB) 的单 GPU 性能测试 - FP8 和 EAGLE 推测解码测试

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--quantization fp8`: FP8 quantization / FP8 量化
- `--speculative-algorithm EAGLE`: EAGLE speculative decoding / EAGLE 推测解码
- `--speculative-num-steps 5`: Speculative steps / 推测步数
- `--speculative-eagle-topk 4`: EAGLE top-k / EAGLE top-k
- `--speculative-num-draft-tokens 16`: Draft tokens count / 草稿令牌数
- `--mem-fraction-static 0.7`: Memory fraction / 内存比例

**Function Points / 功能点**:
- FP8 quantized model offline throughput / FP8 量化模型离线吞吐量
- EAGLE speculative decoding online latency / EAGLE 推测解码在线延迟
- ShareGPT dataset with 3072 context length / 3072 上下文长度的 ShareGPT 数据集

**Observable Points / 可观察点**:
- Output throughput (token/s) / 输出吞吐量 (token/s)
- Median end-to-end latency (ms) / 中位端到端延迟 (ms)
- Accept length for speculative decoding / 推测解码的接受长度
- AMD threshold: > 3500 token/s, < 1800ms latency / AMD 阈值: > 3500 token/s, < 1800ms 延迟
- NVIDIA threshold: > 4300 token/s, < 900ms latency / NVIDIA 阈值: > 4300 token/s, < 900ms 延迟

---

### 3. Bench Serving 1GPU Part 1 / 单 GPU 服务基准测试第一部分 (test_bench_serving_1gpu_part1.py) 📊

**Test Goal / 测试目标**: Performance tests for single GPU LLM throughput/latency and LoRA tests - works on 5090 (32GB) / 单 GPU LLM 吞吐量/延迟和 LoRA 测试 - 适用于 5090 (32GB)

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--disable-radix-cache`: Disable radix cache / 禁用基数缓存
- `--chunked-prefill-size -1`: Disable chunked prefill / 禁用分块预填充
- `--attention-backend triton`: Triton attention backend / Triton 注意力后端
- `--max-running-requests 10`: Limit concurrent requests / 限制并发请求
- `--enable-lora`: Enable LoRA / 启用 LoRA
- `--max-loras-per-batch 1`: Max LoRAs per batch / 每批最大 LoRA 数
- `--lora-paths`: LoRA adapter paths / LoRA 适配器路径

**Function Points / 功能点**:
- Default offline throughput with 500 prompts / 500 个提示的默认离线吞吐量
- Non-streaming small batch size throughput / 非流式小批大小吞吐量
- Throughput without radix cache / 无基数缓存的吞吐量
- Throughput without chunked prefill / 无分块预填充的吞吐量
- Triton attention backend throughput / Triton 注意力后端吞吐量
- Default online latency with 100 prompts / 100 个提示的默认在线延迟
- LoRA online latency tests / LoRA 在线延迟测试
- Concurrent LoRA adapter updates during serving / 服务期间的并发 LoRA 适配器更新

**Observable Points / 可观察点**:
- Output throughput (token/s) / 输出吞吐量 (token/s)
- Median end-to-end latency (ms) / 中位端到端延迟 (ms)
- Median TTFT - Time To First Token (ms) / 中位首令牌时间 (ms)
- Median ITL - Inter-Token Latency (ms) / 中位令牌间延迟 (ms)
- LoRA adapter load/unload success rate / LoRA 适配器加载/卸载成功率

---

### 4. Bench Serving 1GPU Part 2 / 单 GPU 服务基准测试第二部分 (test_bench_serving_1gpu_part2.py) 📊

**Test Goal / 测试目标**: Performance tests for single GPU VLM, Score API, and Embeddings API - works on 5090 (32GB) / 单 GPU VLM、Score API 和 Embeddings API 性能测试 - 适用于 5090 (32GB)

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--mem-fraction-static 0.7`: Memory fraction for VLM / VLM 内存比例
- Dataset: `mmmu` for VLM tests / VLM 测试数据集: mmmu

**Function Points / 功能点**:
- VLM offline throughput with MMMU dataset / 使用 MMMU 数据集的 VLM 离线吞吐量
- VLM online latency with request rate=1 / 请求率=1 的 VLM 在线延迟
- Score API latency and throughput / Score API 延迟和吞吐量
- Score API batch scaling (10, 25, 50) / Score API 批扩展 (10, 25, 50)
- Embeddings API latency and throughput / Embeddings API 延迟和吞吐量
- Embeddings API batch scaling (10, 25, 50) / Embeddings API 批扩展 (10, 25, 50)

**Observable Points / 可观察点**:
- Output throughput (token/s) / 输出吞吐量 (token/s)
- Median end-to-end latency (ms) / 中位端到端延迟 (ms)
- Median TTFT (ms) / 中位首令牌时间 (ms)
- Median ITL (ms) / 中位令牌间延迟 (ms)
- Average latency (ms) / 平均延迟 (ms)
- P95 latency (ms) / P95 延迟 (ms)
- Throughput (req/s) / 吞吐量 (请求/秒)
- Successful requests count / 成功请求数

---

### 5. Bench Serving 2GPU / 2GPU 服务基准测试 (test_bench_serving_2gpu.py) 📊

**Test Goal / 测试目标**: Performance tests for 2-GPU requiring large GPUs (H200 80GB) - MoE and Pipeline Parallel tests / 需要大型 GPU (H200 80GB) 的 2-GPU 性能测试 - MoE 和流水线并行测试

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--tp 2`: Tensor parallel size 2 / 张量并行大小为 2
- `--pp-size 2`: Pipeline parallel size 2 / 流水线并行大小为 2
- `--quantization fp8`: FP8 quantization / FP8 量化
- `--disable-radix-cache`: Disable radix cache / 禁用基数缓存

**Function Points / 功能点**:
- MoE offline throughput with TP=2 / TP=2 的 MoE 离线吞吐量
- MoE throughput without radix cache / 无基数缓存的 MoE 吞吐量
- Pipeline Parallel offline throughput for decode / 用于解码的流水线并行离线吞吐量
- Pipeline Parallel long context prefill (128K tokens) / 流水线并行长上下文预填充 (128K 令牌)

**Observable Points / 可观察点**:
- Output throughput (token/s) / 输出吞吐量 (token/s)
- Input throughput (token/s) for prefill / 预填充的输入吞吐量 (token/s)
- AMD threshold: > 2100 token/s / AMD 阈值: > 2100 token/s
- NVIDIA threshold: > 2200-6700 token/s / NVIDIA 阈值: > 2200-6700 token/s

---

### 6. DeepSeek V3 FP4 4GPU Perf / DeepSeek V3 FP4 4GPU 性能测试 (test_dpsk_v3_fp4_4gpu_perf.py) 🔬

**Test Goal / 测试目标**: Unified performance and accuracy tests for DeepSeek-V3-0324-FP4 on 4x B200 GPUs / 在 4x B200 GPU 上 DeepSeek-V3-0324-FP4 的统一性能和准确性测试

**Test Type / 测试类型**: Performance + Accuracy Test / 性能 + 准确性测试

**Covered Parameters / 覆盖参数**:
- `--tp=4`: Tensor parallel size 4 / 张量并行大小为 4
- `--quantization fp4`: FP4 quantization / FP4 量化
- `--speculative-algorithm=EAGLE`: EAGLE speculative decoding / EAGLE 推测解码
- `--speculative-num-steps=3`: Speculative steps / 推测步数
- `--speculative-eagle-topk=1`: EAGLE top-k / EAGLE top-k
- `--speculative-num-draft-tokens=4`: Draft tokens / 草稿令牌数
- `--trust-remote-code`: Trust remote code / 信任远程代码
- `--model-loader-extra-config`: Multi-threaded loading / 多线程加载

**Function Points / 功能点**:
- DeepSeek-V3-0324-FP4 with TP=4 / TP=4 的 DeepSeek-V3-0324-FP4
- DeepSeek-V3-0324-FP4 with TP=4 + EAGLE speculative decoding / TP=4 + EAGLE 推测解码的 DeepSeek-V3-0324-FP4
- Combined performance and accuracy evaluation / 综合性能和准确性评估
- GSM8K dataset accuracy validation / GSM8K 数据集准确性验证

**Observable Points / 可观察点**:
- Output throughput (token/s) / 输出吞吐量 (token/s)
- Accuracy score (baseline: 0.935) / 准确性分数 (基线: 0.935)
- Performance profiles saved to disk / 保存到磁盘的性能配置文件

---

### 7. GPT OSS 4GPU Perf / GPT OSS 4GPU 性能测试 (test_gpt_oss_4gpu_perf.py) 📊

**Test Goal / 测试目标**: Nightly performance benchmarks for GPT-OSS models on 4x B200 GPUs / 在 4x B200 GPU 上 GPT-OSS 模型的夜间性能基准测试

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--tp 4`: Tensor parallel size 4 / 张量并行大小为 4
- `--cuda-graph-max-bs 200`: CUDA graph max batch size / CUDA 图最大批大小
- `--mem-fraction-static 0.93`: High memory fraction / 高内存比例

**Function Points / 功能点**:
- GPT-OSS-120B model benchmarking / GPT-OSS-120B 模型基准测试
- Variable batch sizes: 1, 8, 16, 64 / 可变批大小: 1, 8, 16, 64
- Fixed input length 4096, output length 512 / 固定输入长度 4096, 输出长度 512
- Nightly benchmark runner with profiling / 带性能分析的夜间基准测试运行器

**Observable Points / 可观察点**:
- Benchmark success/failure status / 基准测试成功/失败状态
- Performance profiles in `performance_profiles_gpt_oss_4gpu` / `performance_profiles_gpt_oss_4gpu` 中的性能配置文件
- Final benchmark report / 最终基准测试报告

---

### 8. Text Models Perf / 文本模型性能测试 (test_text_models_perf.py) 📊

**Test Goal / 测试目标**: Nightly performance benchmarks for text models on 2 GPUs / 在 2 个 GPU 上文本模型的夜间性能基准测试

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- TP=1 for 8B models / 8B 模型的 TP=1
- TP=2 for 57B MoE models / 57B MoE 模型的 TP=2

**Function Points / 功能点**:
- Llama-3.1-8B-Instruct benchmarking / Llama-3.1-8B-Instruct 基准测试
- Qwen2-57B-A14B-Instruct (MoE) benchmarking / Qwen2-57B-A14B-Instruct (MoE) 基准测试
- Variable batch sizes: 1, 8, 16, 64 / 可变批大小: 1, 8, 16, 64
- Configurable input/output lengths via environment / 通过环境变量配置输入/输出长度

**Observable Points / 可观察点**:
- Benchmark results per model / 每个模型的基准测试结果
- Performance profiles in `performance_profiles_text_models` / `performance_profiles_text_models` 中的性能配置文件
- Final aggregated report / 最终汇总报告

---

### 9. VLM Perf 5090 / VLM 性能测试 5090 (test_vlm_perf_5090.py) 📊

**Test Goal / 测试目标**: VLM performance tests optimized for RTX 5090 (32GB) / 针对 RTX 5090 (32GB) 优化的 VLM 性能测试

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--mem-fraction-static 0.7`: Memory fraction / 内存比例
- Dataset: `mmmu` / 数据集: `mmmu`

**Function Points / 功能点**:
- VLM offline throughput with 200 prompts / 200 个提示的 VLM 离线吞吐量
- VLM online latency with 250 prompts at rate=1 / 速率=1 时 250 个提示的 VLM 在线延迟
- Optimized for 5090 GPU constraints / 针对 5090 GPU 约束优化

**Observable Points / 可观察点**:
- Output throughput > 2000 token/s / 输出吞吐量 > 2000 token/s
- Median end-to-end latency < 16500 ms / 中位端到端延迟 < 16500 ms
- Median TTFT < 150 ms / 中位首令牌时间 < 150 ms
- Median ITL < 8 ms / 中位令牌间延迟 < 8 ms

---

### 10. VLMs Perf / VLMs 性能测试 (test_vlms_perf.py) 📊

**Test Goal / 测试目标**: Nightly performance benchmarks for Vision-Language models on 2 GPUs / 在 2 个 GPU 上视觉语言模型的夜间性能基准测试

**Test Type / 测试类型**: Performance Test / 性能测试

**Covered Parameters / 覆盖参数**:
- `--tp=2`: Tensor parallel for 30B models / 30B 模型的张量并行
- `--mem-fraction-static=0.7`: Memory fraction / 内存比例
- `--trust-remote-code`: Trust remote code / 信任远程代码
- `--dataset-name=mmmu`: MMMU dataset / MMMU 数据集

**Function Points / 功能点**:
- Qwen2.5-VL-7B-Instruct benchmarking / Qwen2.5-VL-7B-Instruct 基准测试
- Gemma-3-27b-it benchmarking / Gemma-3-27b-it 基准测试
- Qwen3-VL-30B-A3B-Instruct with TP=2 / TP=2 的 Qwen3-VL-30B-A3B-Instruct
- Variable batch sizes: 1, 2, 8, 16 / 可变批大小: 1, 2, 8, 16
- Configurable via NIGHTLY_VLM_MODELS environment / 可通过 NIGHTLY_VLM_MODELS 环境变量配置

**Observable Points / 可观察点**:
- Benchmark results per VLM model / 每个 VLM 模型的基准测试结果
- Performance profiles in `performance_profiles_vlms` / `performance_profiles_vlms` 中的性能配置文件
- Final benchmark report / 最终基准测试报告

---

## Test File Summary / 测试文件汇总

| # | Test File / 测试文件 | Main Function / 主函数 | Test Type / 测试类型 | Category / 类别 |
|---|---------------------|----------------------|---------------------|----------------|
| 1 | test_bench_one_batch_2gpu.py | MoE TP=2, Torch Compile | Performance / 性能 | Multi-GPU / 多 GPU |
| 2 | test_bench_serving_1gpu_large.py | FP8, EAGLE Speculative | Performance / 性能 | Single GPU Large / 单 GPU 大型 |
| 3 | test_bench_serving_1gpu_part1.py | Throughput, LoRA | Performance / 性能 | Single GPU / 单 GPU |
| 4 | test_bench_serving_1gpu_part2.py | VLM, Score, Embeddings | Performance / 性能 | Single GPU APIs / 单 GPU API |
| 5 | test_bench_serving_2gpu.py | MoE, Pipeline Parallel | Performance / 性能 | Multi-GPU / 多 GPU |
| 6 | test_dpsk_v3_fp4_4gpu_perf.py | DeepSeek V3 FP4 | Performance+Accuracy / 性能+准确性 | 4-GPU Large / 4-GPU 大型 |
| 7 | test_gpt_oss_4gpu_perf.py | GPT-OSS-120B | Performance / 性能 | 4-GPU Nightly / 4-GPU 夜间 |
| 8 | test_text_models_perf.py | Text Models | Performance / 性能 | 2-GPU Nightly / 2-GPU 夜间 |
| 9 | test_vlm_perf_5090.py | VLM 5090 Optimized | Performance / 性能 | Single GPU VLM / 单 GPU VLM |
| 10 | test_vlms_perf.py | VLM Models | Performance / 性能 | 2-GPU Nightly VLM / 2-GPU 夜间 VLM |

---

## Observable Points Summary / 可观察点汇总

### Performance Metrics / 性能指标

| Metric / 指标 | Description / 描述 | Tests / 测试 |
|---------------|-------------------|-------------|
| Output Throughput / 输出吞吐量 | Tokens generated per second / 每秒生成的令牌数 | All serving tests / 所有服务测试 |
| Input Throughput / 输入吞吐量 | Tokens processed per second (prefill) / 每秒处理的令牌数 (预填充) | Long context tests / 长上下文测试 |
| Median E2E Latency / 中位端到端延迟 | End-to-end request latency / 端到端请求延迟 | Online latency tests / 在线延迟测试 |
| Median TTFT / 中位首令牌时间 | Time to first token / 首令牌时间 | Online latency tests / 在线延迟测试 |
| Median ITL / 中位令牌间延迟 | Inter-token latency / 令牌间延迟 | Online latency tests / 在线延迟测试 |
| Accept Length / 接受长度 | Average accepted tokens in speculative decoding / 推测解码中平均接受的令牌数 | EAGLE tests / EAGLE 测试 |

### API Metrics / API 指标

| Metric / 指标 | Description / 描述 | Tests / 测试 |
|---------------|-------------------|-------------|
| Average Latency / 平均延迟 | Mean request latency / 平均请求延迟 | Score, Embeddings API / Score, Embeddings API |
| P95 Latency / P95 延迟 | 95th percentile latency / 第 95 百分位延迟 | Score, Embeddings API / Score, Embeddings API |
| Throughput (req/s) / 吞吐量 (请求/秒) | Requests per second / 每秒请求数 | Score, Embeddings API / Score, Embeddings API |
| Success Rate / 成功率 | Successful requests / total requests / 成功请求/总请求 | All API tests / 所有 API 测试 |

### Accuracy Metrics / 准确性指标

| Metric / 指标 | Description / 描述 | Tests / 测试 |
|---------------|-------------------|-------------|
| Accuracy Score / 准确性分数 | Evaluation dataset accuracy / 评估数据集准确性 | DeepSeek V3 FP4 / DeepSeek V3 FP4 |
| Baseline Comparison / 基线比较 | Against expected baseline / 与预期基线比较 | DeepSeek V3 FP4 / DeepSeek V3 FP4 |

---

## CI/CD Registration / CI/CD 注册

All tests are registered with CI/CD pipelines using the following suites / 所有测试都使用以下套件注册到 CI/CD 管道:

| Test File / 测试文件 | CUDA Suite / CUDA 套件 | AMD Suite / AMD 套件 | Nightly / 夜间 |
|---------------------|----------------------|---------------------|---------------|
| test_bench_one_batch_2gpu.py | stage-b-test-2-gpu-large | stage-b-test-2-gpu-large-amd | No |
| test_bench_serving_1gpu_large.py | stage-b-test-1-gpu-large | stage-b-test-1-gpu-large-amd | No |
| test_bench_serving_1gpu_part1.py | stage-b-test-1-gpu-large | stage-b-test-1-gpu-large-amd | No |
| test_bench_serving_1gpu_part2.py | stage-b-test-1-gpu-large | stage-b-test-1-gpu-large-amd | No |
| test_bench_serving_2gpu.py | stage-b-test-2-gpu-large | stage-b-test-2-gpu-large-amd | No |
| test_dpsk_v3_fp4_4gpu_perf.py | nightly-4-gpu-b200 | - | Yes |
| test_gpt_oss_4gpu_perf.py | nightly-4-gpu-b200 | - | Yes |
| test_text_models_perf.py | nightly-perf-text-2-gpu | - | Yes |
| test_vlm_perf_5090.py | stage-b-test-1-gpu-small | stage-b-test-1-gpu-small-amd | No |
| test_vlms_perf.py | nightly-perf-vlm-2-gpu | - | Yes |

---

## Environment Variables / 环境变量

| Variable / 变量 | Description / 描述 | Used In / 用于 |
|-----------------|-------------------|---------------|
| NIGHTLY_INPUT_LENS / NIGHTLY_INPUT_LENS | Input sequence lengths / 输入序列长度 | test_text_models_perf.py |
| NIGHTLY_OUTPUT_LENS / NIGHTLY_OUTPUT_LENS | Output sequence lengths / 输出序列长度 | test_text_models_perf.py |
| NIGHTLY_VLM_MODELS / NIGHTLY_VLM_MODELS | Custom VLM models list / 自定义 VLM 模型列表 | test_vlms_perf.py |
| NIGHTLY_VLM_BATCH_SIZES / NIGHTLY_VLM_BATCH_SIZES | VLM batch sizes / VLM 批大小 | test_vlms_perf.py |
| NIGHTLY_VLM_INPUT_LENS / NIGHTLY_VLM_INPUT_LENS | VLM input lengths / VLM 输入长度 | test_vlms_perf.py |
| NIGHTLY_VLM_OUTPUT_LENS / NIGHTLY_VLM_OUTPUT_LENS | VLM output lengths / VLM 输出长度 | test_vlms_perf.py |
| SGLANG_ENABLE_SPEC_V2 / SGLANG_ENABLE_SPEC_V2 | Enable speculative V2 / 启用推测 V2 | test_dpsk_v3_fp4_4gpu_perf.py |

---

## Default Models Used / 使用的默认模型

| Model Type / 模型类型 | Default Model / 默认模型 |
|----------------------|-------------------------|
| Standard LLM / 标准 LLM | meta-llama/Llama-3.1-8B-Instruct |
| MoE / MoE | mistralai/Mixtral-8x7B-Instruct-v0.1 |
| FP8 / FP8 | neuralmagic/meta-llama/Llama-3.1-8B-Instruct-FP8 |
| VLM / VLM | openbmb/MiniCPM-V-2_6 |
| Small VLM / 小型 VLM | Qwen/Qwen2.5-VL-3B-Instruct |
| Embedding / 嵌入 | BAAI/bge-large-en-v1.5 |
| Score Model / 评分模型 | lmms-lab/llama3-llava-8b |
| EAGLE Target / EAGLE 目标 | meta-llama/Llama-2-13b-chat-hf |
| EAGLE Draft / EAGLE 草稿 | lmsys/EAGLE-llama2-chat-13B |

---

*Generated by test-case-analyzer-v3 / 由 test-case-analyzer-v3 生成*
