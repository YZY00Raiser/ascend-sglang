# NPU LoRA 测试用例生成文档

本文档列出了根据 `NPU_vs_GPU_LoRA_Test_Gap_Analysis.md` 差距分析报告生成的 NPU LoRA 测试用例。

## 生成文件列表

### 🔴 高优先级测试（8个）

| 序号 | 测试文件 | 测试目标 | 对应GPU测试 |
|-----|---------|---------|-----------|
| 1 | `test_npu_lora_hf_sgl_logprob_diff.py` | HF vs SGLang LoRA 对数概率比较 | `test_lora_hf_sgl_logprob_diff.py` |
| 2 | `test_npu_lora_eviction.py` | LoRA 适配器驱逐行为 | `test_lora_eviction.py` |
| 3 | `test_npu_lora_update.py` | 动态适配器加载/卸载 | `test_lora_update.py` |
| 4 | `test_npu_multi_lora_backend.py` | 多 LoRA 批次处理 | `test_multi_lora_backend.py` |
| 5 | `test_npu_lora_tp.py` | 张量并行 LoRA | `test_lora_tp.py` |
| 6 | `test_npu_embedding_lora_support.py` | 嵌入模型 LoRA | `test_embedding_lora_support.py` |
| 7 | `test_npu_lora_moe_tp_logprob_diff.py` | MoE TP 一致性 | `test_lora_moe_tp_logprob_diff.py` |
| 8 | `test_npu_lora_qwen3.py` | Qwen3 基础 LoRA | `test_lora_qwen3.py` |

### 🟡 中优先级测试（部分生成）

| 序号 | 测试文件 | 测试目标 | 对应GPU测试 |
|-----|---------|---------|-----------|
| 9 | `test_npu_lora_radix_cache.py` | Radix 缓存与 LoRA | `test_lora_radix_cache.py` |
| 10 | `test_npu_lora_tied_lm_head.py` | 绑定 LM Head LoRA | `test_lora_tied_lm_head.py` |
| 11 | `test_npu_lora_qwen3_5_4b_logprob_diff.py` | Qwen3.5-4B LoRA 精度 | `test_lora_qwen3_5_4b_logprob_diff.py` |
| 12 | `test_npu_lora_deepseek_v3_base_logprob_diff.py` | DeepSeek-V3 MLA LoRA 精度 | `test_lora_deepseek_v3_base_logprob_diff.py` |

## 测试用例结构说明

每个测试文件遵循以下结构：

```python
"""
测试描述 - 测试目标说明
"""

import ...

# 测试配置
MODEL_NAME = "..."
ADAPTER_URL = "..."

class TestNpuLoRA...:
    """测试类描述"""

    @pytest.fixture(scope="class")
    def server_process(self):
        """启动SGLang服务器"""
        ...

    def test_xxx(self, server_process):
        """测试方法描述"""
        ...
```

## 运行测试

### 运行所有生成的测试

```bash
cd d:\skill_test\sglang\test\registered\ascend\basic_function\lora\generated_tests
pytest -v
```

### 运行特定测试

```bash
# 运行特定测试文件
pytest test_npu_lora_hf_sgl_logprob_diff.py -v

# 运行特定测试方法
pytest test_npu_lora_tp.py::TestNpuLoRATensorParallel::test_tp2_lora_inference -v
```

### 并行运行测试（需要多GPU/NPU）

```bash
pytest -n auto -v
```

## 注意事项

1. **硬件要求**: 部分测试（如TP=2测试）需要多个NPU
2. **模型下载**: 首次运行会自动下载模型，可能需要较长时间
3. **内存要求**: 大型模型测试需要足够的NPU内存
4. **网络要求**: 需要从HuggingFace下载模型和适配器

## 测试覆盖范围

### 高优先级测试覆盖
- ✅ HF vs SGLang 精度对比
- ✅ LoRA 驱逐行为
- ✅ 动态适配器更新
- ✅ 多 LoRA 批次处理
- ✅ 张量并行 LoRA
- ✅ 嵌入模型 LoRA
- ✅ MoE TP 一致性
- ✅ Qwen3 基础 LoRA

### 中优先级测试覆盖
- ✅ Radix 缓存与 LoRA
- ✅ 绑定 LM Head LoRA
- ✅ Qwen3.5-4B LoRA 精度
- ✅ DeepSeek-V3 MLA LoRA 精度

---

**生成日期**: 2026-05-12
**作者**: AI Assistant
**基于**: NPU_vs_GPU_LoRA_Test_Gap_Analysis.md
