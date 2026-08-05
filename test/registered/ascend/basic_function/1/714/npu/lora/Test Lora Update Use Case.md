# Test Lora Update Use Case

> **Workspace:** d:\sglang

---

**User**

`d:\sglang\test\registered\lora\test_lora_update.py` 该用例在测试什么

**Assistant**

该用例测试 **SGLang 在推理服务运行期间对 LoRA adapter 进行动态加载、卸载、推理的能力**，以及动态加载与静态加载输出的等价性。

核心测试内容：

1. **动态 LoRA 生命周期管理**
   - 通过 `/load_lora_adapter`、`/unload_lora_adapter` 在运行时加载/卸载 LoRA
   - 支持多种 lora_paths 初始化格式（字符串、`name=path`、dict 含 `pinned` 等）

2. **错误处理**
   - 重复加载返回 `already loaded`
   - 未加载就推理返回 `never been loaded`
   - adapter 与当前 `max-lora-rank` / `lora-target-modules` 不兼容返回 `incompatible`
   - 因 pinned adapter 占满槽位导致新加载失败返回 `starvation`

3. **`max_lora_rank` 与 `lora_target_modules`**
   - 显式指定 rank、target modules 时的兼容性行为
   - 从初始 adapter 隐式推断这些参数时的行为

4. **`max_loaded_loras` 与隐式驱逐（implicit eviction）**
   - 当加载数量超过上限时，LRU 非 pinned adapter 会被自动驱逐
   - 当前 forward pass 需要的 adapter 不会被驱逐
   - pinned adapter 不会被驱逐，且可能因占满槽位导致 starvation

5. **动态 vs 静态加载输出一致性**
   - 同一批 forward 操作，先动态 load/unload 后再推理
   - 与一开始就把所有 adapter 静态加载后的推理输出
   - 用 ROUGE-L ≥ 0.9 判断两者输出一致

6. **`/v1/models` 接口**
   - 验证返回列表会随加载/卸载 adapter 动态变化
   - base model 的 `parent` 为 null，adapter 的 `parent` 指向 base model

当前实际执行的测试方法主要是：
- `test_dynamic_lora_update_server`：server 模式下跑上述全部操作序列
- `test_v1_models_endpoint_with_lora`：验证 `/v1/models` 对 LoRA 的曝光

engine 模式的方法被注释掉了，目前 server 模式是主要执行路径。

---

**User**

解释具体代码行

**Assistant**

你想让我解释哪些具体代码行？可以告诉我：

- 行号范围，例如 `100-150`
- 某个函数/类名，例如 `_run_operation_sequence`
- 或者某个你困惑的代码片段

我可以针对指定部分逐行解释。

---

**User**

逐行分析所有

**Assistant**

文件较长（1558 行），下面按**逻辑块**逐段解释。每个段落会给出对应行号与文件链接。

---

## 1. 文件头部与导入（[行 1–46](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1-L46)）

```python
1-16   # Copyright / Apache License 头
17     import json                    # Server 模式下发请求、序列化 adapter dict 用
18     import multiprocessing as mp   # 入口 __main__ 中设置 spawn 启动方法
19     import unittest                # 单元测试框架
20     from dataclasses import dataclass
21     from enum import Enum
22     from typing import Any, Iterable, List, Optional, Union
23
24     import requests                # Server 模式调用 HTTP API
25     import torch
26
27     from sglang.srt.utils import kill_process_tree
28     from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
29     from sglang.srt.utils import kill_process_tree
30     from sglang.test.runners import SRTRunner
31     from sglang.test.test_utils import (
32         DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
33         DEFAULT_URL_FOR_TEST,
34         CustomTestCase,
35         calculate_rouge_l,
36         is_in_ci,
37         popen_launch_server,
38     )
```

- `SRTRunner`：进程内 engine 模式运行器。
- `popen_launch_server` / `kill_process_tree`：Server 模式启停子进程。
- `calculate_rouge_l`：用于对比动态加载与静态加载的输出文本。
- `is_in_ci`：判断是否在 CI 环境，决定跑全部还是只跑 BASIC。
- `register_cuda_ci` / `register_amd_ci`：给 CI 系统注册该测试的预计耗时与运行配置。

---

## 2. CI 注册与常量（[行 39–56](file:///d:/sglang/test/registered/lora/test_lora_update.py#L39-L56)）

```python
39-46  register_cuda_ci(est_time=487, stage="base-b", runner_config="1-gpu-large")
47-54  register_amd_ci(est_time=730, suite="stage-b-test-1-gpu-large-amd")
55
56     PROMPTS = [
57         "SGL is a",
58         "AI is a field of computer science focused on",
59         "Computer science is the study of",
60         "Write a short story.",
61         "What are the main components of a computer?",
62     ]
63
64     MEM_FRACTION_STATIC = 0.8
```

- `PROMPTS`：所有 forward 操作复用的 5 条提示词。
- `MEM_FRACTION_STATIC`：Server/Engine 启动时分配给 KV cache 等的静态显存比例。

---

## 3. 核心数据结构（[行 58–96](file:///d:/sglang/test/registered/lora/test_lora_update.py#L58-L96)）

```python
58     class OperationType(Enum):
59         LOAD = "load"
60         UNLOAD = "unload"
61         FORWARD = "forward"
```

定义三种操作类型：加载、卸载、前向推理。

```python
64     @dataclass
65     class Operation:
66         type: OperationType
67         data: Optional[Any]
68         expected_error: Optional[str] = None
69         expected_implicit_evictions: Optional[set[str]] = None
```

- `type`：操作类型。
- `data`：操作数据。LOAD/UNLOAD 为 adapter 名或 dict；FORWARD 为 `(prompt, adapter)` 列表。
- `expected_error`：期望该操作失败，并检查错误消息包含该子串。
- `expected_implicit_evictions`：由于 `max_loaded_loras` 限制，某些 adapter 可能被隐式驱逐，这里记录预期被驱逐的集合，用于校验 `loaded_adapters`。

```python
77     @dataclass
78     class TestCase:
79         description: str
80         base: str
81         max_loras_per_batch: int
82         all_adapters: List[str]
83         op_sequence: List[Operation]
84         initial_adapters: Optional[List[str]] = None
85         enable_lora: Optional[bool] = None
86         max_lora_rank: Optional[int] = None
87         lora_target_modules: Optional[List] = None
88         max_new_tokens: int = 32
89         max_loaded_loras: Optional[int] = None
```

- `base`：基础模型路径，如 `meta-llama/Llama-3.1-8B-Instruct`。
- `all_adapters`：该用例涉及的所有 adapter，用于“静态 pass”一次性全加载。
- `op_sequence`：动态加载/卸载/推理的操作序列。
- `initial_adapters`：Server/Engine 启动时已加载的 adapter。

```python
92     def create_batch_data(adapters):
93         if not isinstance(adapters, list):
94             adapters = [adapters]
95         return [(prompt, adapter) for prompt in PROMPTS for adapter in adapters]
```

把 `PROMPTS` 与给定的 adapter 列表做笛卡尔积，生成 `(prompt, lora_path)` 对。`adapter` 可为 `None`（表示使用 base 模型推理）。

## 4. BASIC_TESTS（[行 98–339](file:///d:/sglang/test/registered/lora/test_lora_update.py#L98-L339)）

该列表包含两个基础测试用例。

### 用例 1：带初始 lora_paths 的动态更新（行 98–209）

```python
98-105   TestCase(
99           description="dynamic lora update with initial lora_paths",
100          base="meta-llama/Llama-3.1-8B-Instruct",
101          max_loras_per_batch=3,
102          all_adapters=[...3 个 HuggingFace adapter...],
103          initial_adapters=[
104              # 测试 3 种支持的 lora-path 格式
105              "philschmid/code-llama-3-1-8b-text-to-sql-lora",
106              "nvidia/llama-3.1-nemoguard-8b-topic-control=nvidia/...",
107              {
108                  "lora_name": "pbevan11/llama-3.1-8b-ocr-correction",
109                  "lora_path": "pbevan11/llama-3.1-8b-ocr-correction",
110                  "pinned": False,
111              },
112          ],
113          op_sequence=[...]
```

`initial_adapters` 同时验证三种初始化格式：
- 纯字符串（name 与 path 相同）
- `name=path` 形式
- dict 形式（可带 `pinned` 等字段）

后续 `op_sequence` 依次执行：
1. 重复加载已存在的 adapter → 期望 `already loaded`
2. 卸载后再重新加载
3. 对 3 个 adapter 做 batch forward
4. 卸载两个 adapter后，分别对已卸载/未加载的 adapter 做 forward → 期望失败
5. 再次加载后做 batch forward
6. 全部卸载后再 forward → 验证 base 模型也能跑（`None`）

### 用例 2：不带初始 lora_paths 的动态更新（行 210–339）

```python
210      TestCase(
211          description="dynamic lora update without initial lora_paths",
212          base="meta-llama/Llama-3.1-8B-Instruct",
213          enable_lora=True,
214          max_lora_rank=256,
215          lora_target_modules=["all"],
216          max_loras_per_batch=4,
```

这里显式开启 `enable_lora`、指定最大 rank 256、target modules 为 `all`。操作序列与用例 1 类似，但在 forward 中混入 `None`（base 模型），验证 base + LoRA 混合 batch。

---

## 5. TARGET_MODULE_TESTS（[行 340–456](file:///d:/sglang/test/registered/lora/test_lora_update.py#L340-L456)）

测试 `lora_target_modules` 的兼容性问题。

- 用例 1：显式指定 `lora_target_modules=["all"]`，可以加载 target modules 更多的 adapter。
- 用例 2：不指定 `lora_target_modules`，从第一个加载的较大 adapter 隐式推断，仍可加载较小的 adapter。
- 用例 3：从较小的 adapter 推断 `lora_target_modules`，后续加载 target modules 更多的 adapter 会报 `incompatible` 错误。

---

## 6. MAX_LORA_RANK_TESTS（[行 457–569](file:///d:/sglang/test/registered/lora/test_lora_update.py#L457-L569)）

测试 `max_lora_rank` 限制。

- 用例 1：显式 `max_lora_rank=32`，只能加载 rank ≤32 的 adapter；rank=256 的会报 `incompatible`。
- 用例 2：不指定 `max_lora_rank`，从初始 adapter 隐式推断；若初始 adapter rank=32，则 rank=256 的同样无法加载。

---

## 7. MAX_LOADED_LORAS_TESTS（[行 570–730](file:///d:/sglang/test/registered/lora/test_lora_update.py#L570-L730)）

测试 `max_loaded_loras` 限制及隐式驱逐策略。

### 用例 1：隐式驱逐与重新加载（行 570–659）

```python
570      TestCase(
571          description="Test max_loaded_loras limit as well as implicit eviction and reloading",
572          base="...",
573          max_loras_per_batch=2,
574          max_loaded_loras=2,
```

- 加载第 3 个 adapter 时，最早加载的非 pinned adapter 被隐式驱逐。
- Forward pass 需要的 adapter 即使 LRU 也不会被驱逐。
- 通过 `expected_implicit_evictions` 校验驱逐集合。

### 用例 2：pinned adapter 的隐式驱逐（行 660–730）

```python
660      TestCase(
661          description="Test implicit eviction and reloading with pinned LoRA adapters",
```

- pinned adapter 不会被驱逐，导致非 pinned adapter 更容易被驱逐。
- 若所有槽位都被 pinned 占满，新加载会报 `starvation`。
- 先卸载 pinned adapter 后，才能重新加载新的 pinned adapter。

---

## 8. EVICTION_TESTS（[行 731–853](file:///d:/sglang/test/registered/lora/test_lora_update.py#L731-L853)）

更复杂的驱逐场景，使用自定义 `lora_name`（`lora1=...`、`lora2=...`）。

- 加载 pinned adapter 导致 starvation。
- 切换 pinned/unpinned 状态。
- 验证不同组合 forward 后，哪些 adapter 仍驻留内存。

## 9. ALL_TESTS 与会话模式定义（[行 854–866](file:///d:/sglang/test/registered/lora/test_lora_update.py#L854-L866)）

```python
854      ALL_TESTS = (
855          BASIC_TESTS
856          + TARGET_MODULE_TESTS
857          + MAX_LORA_RANK_TESTS
858          + MAX_LOADED_LORAS_TESTS
859          + EVICTION_TESTS
860      )
861
862      class LoRAUpdateTestSessionMode(Enum):
863          ENGINE = "engine"
864          SERVER = "server"
```

- `ALL_TESTS`：非 CI 环境跑全部用例；CI 环境只跑 `BASIC_TESTS`。
- `ENGINE` / `SERVER`：两种测试模式。

---

## 10. LoRAUpdateTestSessionBase（[行 868–951](file:///d:/sglang/test/registered/lora/test_lora_update.py#L868-L951)）

```python
868      class LoRAUpdateTestSessionBase:
```

上下文管理器基类，封装测试会话的创建与销毁。

```python
869-900  def __init__(...):
```

保存所有启动参数：模型路径、adapter 列表、`max_loras_per_batch`、`max_loaded_loras`、`max_lora_rank`、`lora_target_modules`、`enable_lora` 等。

```python
901-915  # 解析 initial_adapters，构建 expected_adapters
916      self.expected_adapters = set()
917      if self.lora_paths:
918          for adapter in self.lora_paths:
919              if isinstance(adapter, dict):
920                  lora_name = adapter["lora_name"]
921              elif "=" in adapter:
922                  lora_name = adapter.split("=")[0]
923              else:
924                  lora_name = adapter
925              self.expected_adapters.add(lora_name)
```

根据启动时传入的 adapter 初始化“期望已加载集合”，后续 load/unload 时维护该集合，用于和返回的 `loaded_adapters` 对比。

```python
927-951  # __enter__ / __exit__ 与抽象方法 load/unload/forward
```

子类必须实现：
- `load_lora_adapter`
- `unload_lora_adapter`
- `forward`

---

## 11. LoRAUpdateEngineTestSession（[行 952–1100](file:///d:/sglang/test/registered/lora/test_lora_update.py#L952-L1100)）

进程内 engine 模式。

```python
952      class LoRAUpdateEngineTestSession(LoRAUpdateTestSessionBase):
```

### `__enter__`（行 963–988）

```python
963      def __enter__(self):
964          self.handle = SRTRunner(
965              model_path=self.model_path,
966              model_type="generation",
967              lora_paths=self.lora_paths,
968              max_lora_rank=self.max_lora_rank,
969              lora_target_modules=self.lora_target_modules,
970              lora_backend=self.lora_backend,
971              torch_dtype=torch.float16,
972              mem_fraction_static=MEM_FRACTION_STATIC,
973              max_loras_per_batch=self.max_loras_per_batch,
974              max_loaded_loras=self.max_loaded_loras,
975              disable_cuda_graph=self.disable_cuda_graph,
976              cuda_graph_max_bs_decode=self.cuda_graph_max_bs_decode,
977              enable_lora=self.enable_lora,
978              disable_radix_cache=True,
979          )
980          self.handle.__enter__()
```

使用 `SRTRunner` 在进程内启动 engine，并传入所有 LoRA 相关参数。

### `load_lora_adapter`（行 989–1029）

```python
989      def load_lora_adapter(self, lora_name, lora_path=None, expected_error=None,
990                            pinned=False, expected_implicit_evictions=None):
```

- 调用 `self.handle.load_lora_adapter(...)`。
- 若 `expected_error` 非空，断言返回失败且错误消息包含子串。
- 否则更新 `self.expected_adapters`，并校验返回的 `loaded_adapters` 与期望集合一致。
- 若给出 `expected_implicit_evictions`，从期望集合中扣除被驱逐的 adapter。

### `unload_lora_adapter`（行 1031–1048）

- 先从 `self.expected_adapters` 移除。
- 调用 `self.handle.unload_lora_adapter(...)`。
- 校验返回的 `loaded_adapters`。

### `forward`（行 1050–1098）

- 调用 `self.handle.batch_forward(...)`。
- 捕获 `ValueError` 以处理 `expected_error`。
- 否则校验输出长度，并更新 `expected_adapters`（forward 中使用的 adapter 可能被隐式加载）。
- 同样处理 `expected_implicit_evictions`。

---

## 12. LoRAUpdateServerTestSession（[行 1101–1282](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1101-L1282)）

独立 Server 子进程模式。

### `__enter__`（行 1112–1154）

```python
1112     def __enter__(self):
1113         other_args = [
1114             "--cuda-graph-max-bs-decode", str(self.cuda_graph_max_bs_decode),
1115             "--max-loras-per-batch", str(self.max_loras_per_batch),
1116             "--lora-backend", self.lora_backend,
1117             "--random-seed", "42",
1118             "--max-running-request", "1",
1119             "--mem-fraction-static", str(MEM_FRACTION_STATIC),
1120             "--disable-radix-cache",
1121         ]
```

构造 sglang server 的启动参数。随后根据条件追加：
- `--enable-lora`
- `--lora-paths` 及初始 adapter（dict 会 `json.dumps`）
- `--disable-cuda-graph`
- `--max-lora-rank`
- `--lora-target-modules`
- `--max-loaded-loras`

```python
1148     self.handle = popen_launch_server(
1149         self.model_path,
1150         DEFAULT_URL_FOR_TEST,
1151         DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
1152         other_args=other_args,
1153     )
```

启动独立 server 进程。

### `load_lora_adapter`（行 1155–1198）

```python
1155     def load_lora_adapter(self, ...):
1156         ...
1157         response = requests.post(
1158             DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
1159             json={"lora_name": lora_name, "lora_path": lora_path, "pinned": pinned},
1160         )
```

通过 HTTP POST `/load_lora_adapter` 加载。错误校验与 engine 模式类似，但检查 HTTP 400。

### `unload_lora_adapter`（行 1200–1222）

POST `/unload_lora_adapter`。

### `forward`（行 1224–1282）

POST `/generate`，传入：
```python
{
    "text": prompts,
    "lora_path": lora_paths,
    "sampling_params": {
        "temperature": 0,
        "top_k": 1,
        "max_new_tokens": max_new_tokens,
    },
}
```

同样校验错误、输出长度、`expected_implicit_evictions`。

## 13. 会话工厂（[行 1283–1295](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1283-L1295)）

```python
1283     def LoRAUpdateTestSession(testcase, mode, **kwargs):
1284         if mode == LoRAUpdateTestSessionMode.ENGINE:
1285             return LoRAUpdateEngineTestSession(testcase=testcase, **kwargs)
1286         elif mode == LoRAUpdateTestSessionMode.SERVER:
1287             return LoRAUpdateServerTestSession(testcase=testcase, **kwargs)
1288         else:
1289             raise ValueError(f"Unrecognized mode: {mode!r}")
```

根据模式返回 engine 或 server 会话实例。

---

## 14. TestLoRADynamicUpdate（[行 1296–1500+](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1296-L1500)）

```python
1296     class TestLoRADynamicUpdate(CustomTestCase):
1297         """
1298         This test case verifies that the SRT runner can dynamically load and unload LoRA adapters
1299         during a sequence of operations, and that the outputs of forward passes with dynamically loaded
1300         adapters match the outputs of forward passes with statically loaded adapters.
1301         """
```

类注释说明核心目标：验证动态加载/卸载能力，以及动态与静态加载输出一致。

### `_run_operation_sequence`（行 1308–1375）

```python
1308     def _run_operation_sequence(
1309         self,
1310         mode: LoRAUpdateTestSessionMode,
1311         base: str,
1312         initial_adapters: List[Union[str, dict]],
1313         op_sequence: List[Operation],
1314         max_loras_per_batch: int,
1315         ...
1316     ) -> List[tuple]:
```

执行一组操作序列，返回所有成功 forward 的输出。

```python
1325     forward_outputs = []
1326     with LoRAUpdateTestSession(...) as session:
1327         for op in op_sequence:
1328             ...
1329             if op_type == OperationType.LOAD:
1330                 ...
1331             elif op_type == OperationType.UNLOAD:
1332                 ...
1333             elif op_type == OperationType.FORWARD:
1334                 prompts, adapters = zip(*data)
1335                 result = session.forward(...)
1336                 if not expected_error:
1337                     forward_outputs.append(result)
```

- LOAD：把字符串转换为 dict，调用 `session.load_lora_adapter`。
- UNLOAD：调用 `session.unload_lora_adapter`。
- FORWARD：解包 `(prompt, adapter)`，调用 `session.forward`，无错误时收集输出。

### `_run_dynamic_adapter_updates`（行 1377–1450）

```python
1377     def _run_dynamic_adapter_updates(self, mode, test_cases):
```

对每个 `TestCase` 执行两遍：

1. **Dynamic pass**：按 `op_sequence` 逐步 load/unload/forward。
2. **Static pass**：把所有 `all_adapters` 在启动时加载，只执行 forward 操作。

然后对比两遍输出：

```python
1445     ROUGE_L_TOL = 0.9
1446     ...
1447     self.assertEqual(len(dynamic_output), len(static_output))
1448     for i, (dynamic, static) in enumerate(zip(dynamic_output, static_output), start=1):
1449         self.assertEqual(len(dynamic), len(static))
1450         for j, (d_out, s_out) in enumerate(zip(dynamic, static), start=1):
1451             rouge_score = calculate_rouge_l([d_out_str], [s_out_str])[0]
1452             self.assertGreaterEqual(rouge_score, ROUGE_L_TOL, ...)
```

要求 ROUGE-L 分数 ≥ 0.9。

### `test_dynamic_lora_update_server`（行 1472–1480）

```python
1472     def test_dynamic_lora_update_server(self):
1473         test_cases = BASIC_TESTS if is_in_ci() else ALL_TESTS
1474         self._run_dynamic_adapter_updates(
1475             mode=LoRAUpdateTestSessionMode.SERVER, test_cases=test_cases
1476         )
```

当前实际执行的 server 模式测试。

### `test_dynamic_lora_update_engine`（行 1463–1470）

被注释掉了，未启用。

---

## 15. `test_v1_models_endpoint_with_lora`（[行 1482–1547](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1482-L1547)）

验证 OpenAI 兼容的 `/v1/models` 接口。

```python
1482     def test_v1_models_endpoint_with_lora(self):
1483         adapters = [...]
1484         with LoRAUpdateTestSession(...) as session:
1485             # 无 adapter
1486             response = requests.get(DEFAULT_URL_FOR_TEST + "/v1/models")
1487             self.assertEqual(len(models_data["data"]), 1)  # 只有 base model
1488             ...
1489             # 加载 adapter1
1490             session.load_lora_adapter(lora_name="adapter1", lora_path=adapters[0])
1491             response = requests.get(...)
1492             self.assertEqual(len(models_data["data"]), 2)
1493             ...
1494             # 加载 adapter2
1495             session.load_lora_adapter(lora_name="adapter2", lora_path=adapters[1])
1496             self.assertEqual(len(models_data["data"]), 3)
1497             ...
1498             # 卸载 adapter1
1499             session.unload_lora_adapter(lora_name="adapter1")
1500             self.assertEqual(len(models_data["data"]), 2)
```

- 检查返回列表数量随加载/卸载变化。
- 检查 adapter 的 `id`、`root`、`parent` 字段。

---

## 16. 主入口（[行 1550–1558](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1550-L1558)）

```python
1550     if __name__ == "__main__":
1551         try:
1552             mp.set_start_method("spawn")
1553         except RuntimeError:
1554             pass
1555
1556         unittest.main(warnings="ignore")
```

- 设置多进程启动方式为 `spawn`，避免 CUDA fork 问题。
- 运行 unittest。

以上是对 `test_lora_update.py` 的完整逐段解析。简言之，该文件通过声明式 `TestCase` + `Operation` 序列，覆盖了 SGLang 动态 LoRA 加载/卸载、兼容性检查、隐式驱逐、pinned 策略、以及动态/静态输出一致性等核心场景，最终通过 server HTTP API 实际执行验证。

---

**User**

源码与解释对应

**Assistant**

下面是 `test_lora_tied_lm_head.py` 的源码与解释逐段对照。

---

## 1. 模块注释（[行 14–34](file:///d:/sglang/test/registered/lora/test_lora_tied_lm_head.py#L14-L34)）

```python
"""
Test LoRA on models with tied lm_head (tie_word_embeddings=True).

When tie_word_embeddings=True, lm_head shares the same weight tensor as
embed_tokens. PyTorch's named_modules() deduplicates by object identity,
so lm_head won't appear as a separate module. This test validates that
SGLang correctly handles this case by untying lm_head before LoRA wrapping.

The test:
1. Programmatically creates a LoRA adapter with lm_head in target_modules
   using PEFT on a model with tie_word_embeddings=True (Qwen/Qwen2.5-0.5B).
2. Compares logprobs between HuggingFace+PEFT and SGLang to ensure numerical
   consistency. This implicitly verifies no NaN values are produced and that
   LoRA is actually being applied (since HF+PEFT is the trusted reference).
"""
```

**解释**：
- 测试目标：验证 SGLang 在 `tie_word_embeddings=True` 的模型上，对 `lm_head` 应用 LoRA 的正确性。
- 背景问题：当词嵌入与输出头共享权重时，`lm_head` 不会作为独立模块出现在 `named_modules()` 中，SGLang 需要特殊处理（通常是先解绑再包装 LoRA）。
- 验证方式：用 PEFT 生成一个只 target `lm_head` 的 adapter，然后对比 HuggingFace+PEFT 与 SGLang 的 logprob 差异。

---

## 2. 导入与 CI 注册（[行 36–58](file:///d:/sglang/test/registered/lora/test_lora_tied_lm_head.py#L36-L58)）

```python
import multiprocessing as mp
import os
import shutil
import tempfile
import unittest

import torch

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "peft", "--no-deps"])
    from peft import LoraConfig, get_peft_model

from transformers import AutoModelForCausalLM

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import DEFAULT_PORT_FOR_SRT_TEST_RUNNER, CustomTestCase
```

**解释**：
- `peft`：若未安装则自动 `pip install`。
- `HFRunner` / `SRTRunner`：分别运行 HuggingFace+PEFT 参考实现和 SGLang 实现。
- `register_cuda_ci`：将该测试注册到 CUDA CI 的 `nightly-1-gpu` suite。

---

## 3. 常量配置（[行 60–71](file:///d:/sglang/test/registered/lora/test_lora_tied_lm_head.py#L60-L71)）

```python
register_cuda_ci(est_time=120, suite="nightly-1-gpu", nightly=True)

BASE_MODEL = "Qwen/Qwen2.5-0.5B"

TEST_PROMPTS = [
    "AI is a field of computer science focused on",
    "The capital of France is",
]

MAX_NEW_TOKENS = 16
LOGPROB_THRESHOLD = 2e-1
```

**解释**：
- `BASE_MODEL`：选用 `Qwen2.5-0.5B`，该模型 `tie_word_embeddings=True`。
- `LOGPROB_THRESHOLD = 2e-1`：SGLang 与 HF 的 logprob 最大允许差异。

---

## 4. 创建 LoRA Adapter（[行 73–128](file:///d:/sglang/test/registered/lora/test_lora_tied_lm_head.py#L73-L128)）

```python
def create_lora_adapter_with_lm_head(base_model_name: str, output_dir: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
```

**解释**：加载基础模型到 CPU，使用 fp16。

```python
    assert (
        model.config.tie_word_embeddings
    ), f"Expected tie_word_embeddings=True for {base_model_name}"
```

**解释**：断言基础模型确实使用了 tied embeddings。

```python
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["lm_head"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)
```

**解释**：用 PEFT 创建只作用于 `lm_head` 的 LoRA adapter。

```python
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
```

**解释**：PEFT 默认把 `lora_B` 初始化为 0，这样 adapter 输出与 base 模型相同。这里手动把 `lora_B` 初始化为非零随机值，确保 adapter 有实际影响。

```python
    peft_model.save_pretrained(output_dir)
```

**解释**：保存 adapter 到临时目录。

```python
    from safetensors import safe_open

    safetensors_path = os.path.join(output_dir, "adapter_model.safetensors")
    f = safe_open(safetensors_path, framework="pt")
    lm_head_keys = [k for k in f.keys() if "lm_head" in k]
    assert len(lm_head_keys) > 0, ...
```

**解释**：读取保存的 safetensors，验证其中确实包含 `lm_head` 的 LoRA 权重。

```python
    del peft_model, model
    torch.cuda.empty_cache()
```

**解释**：释放模型占用的内存，为后续 SGLang/HF 运行腾出空间。

---

## 5. 测试类生命周期（[行 130–159](file:///d:/sglang/test/registered/lora/test_lora_tied_lm_head.py#L130-L159)）

```python
class TestLoRATiedLMHead(CustomTestCase):

    _adapter_dir = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._adapter_dir = tempfile.mkdtemp(prefix="sglang_test_lora_tied_lm_head_")
        create_lora_adapter_with_lm_head(BASE_MODEL, cls._adapter_dir)

    @classmethod
    def tearDownClass(cls):
        if cls._adapter_dir and os.path.exists(cls._adapter_dir):
            shutil.rmtree(cls._adapter_dir)
        super().tearDownClass()
```

**解释**：
- `setUpClass`：整个测试类开始前，创建临时 adapter 目录并生成 LoRA adapter。
- `tearDownClass`：测试类结束后删除临时目录。

---

## 6. 核心测试方法（[行 161–224](file:///d:/sglang/test/registered/lora/test_lora_tied_lm_head.py#L161-L224)）

```python
    def test_tied_lm_head_lora_hf_sgl_logprob_match(self):
        prompts = TEST_PROMPTS[:2]
```

**解释**：取两条测试 prompt。

### 6.1 SGLang 运行

```python
        with SRTRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
            lora_paths=[self._adapter_dir],
            max_loras_per_batch=1,
            lora_backend="triton",
            lora_target_modules=["lm_head"],
            disable_cuda_graph=True,
            disable_radix_cache=True,
            mem_fraction_static=0.80,
            port=DEFAULT_PORT_FOR_SRT_TEST_RUNNER,
        ) as srt_runner:
            srt_outputs = srt_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[self._adapter_dir] * len(prompts),
            )
```

**解释**：启动 SGLang engine，加载刚生成的 adapter，并执行 forward。

### 6.2 HuggingFace 运行

```python
        torch.cuda.empty_cache()

        with HFRunner(
            BASE_MODEL,
            torch_dtype=torch.float16,
            model_type="generation",
        ) as hf_runner:
            hf_outputs = hf_runner.forward(
                prompts,
                max_new_tokens=MAX_NEW_TOKENS,
                lora_paths=[self._adapter_dir] * len(prompts),
            )
```

**解释**：释放显存后，用 HuggingFace+PEFT 运行同样的 prompt 和 adapter，作为参考。

### 6.3 Prefill logprob 对比

```python
        for i in range(len(prompts)):
            srt_logprobs = torch.tensor(srt_outputs.top_input_logprobs[i])
            hf_logprobs = torch.tensor(hf_outputs.top_input_logprobs[i])
            max_diff = torch.max(torch.abs(srt_logprobs - hf_logprobs)).item()
            print(f"Prompt {i} prefill logprob max_diff (SGLang vs HF): {max_diff:.6e}")
            self.assertLess(max_diff, LOGPROB_THRESHOLD, ...)
```

**解释**：对比 prefill 阶段每个 token 的 top logprob，最大差异必须小于 `2e-1`。

### 6.4 Decode logprob 对比

```python
        for i in range(len(prompts)):
            srt_logprobs = torch.tensor(srt_outputs.top_output_logprobs[i])
            hf_logprobs = torch.tensor(hf_outputs.top_output_logprobs[i])
            max_diff = torch.max(torch.abs(srt_logprobs - hf_logprobs)).item()
            print(f"Prompt {i} decode logprob max_diff (SGLang vs HF): {max_diff:.6e}")
            self.assertLess(max_diff, LOGPROB_THRESHOLD, ...)
```

**解释**：对比 decode 生成阶段每个 token 的 top logprob，差异同样要小于阈值。

---

## 7. 主入口（[行 226–234](file:///d:/sglang/test/registered/lora/test_lora_tied_lm_head.py#L226-L234)）

```python
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
```

**解释**：设置多进程启动方式为 `spawn`，然后运行 unittest。

这个文件的核心目的是：**验证当基础模型 `tie_word_embeddings=True` 时，SGLang 仍能对共享权重的 `lm_head` 正确应用 LoRA，并且数值结果与 HuggingFace+PEFT 一致**。

---

**User**

我说的是test_lora_update.py

**Assistant**

下面是 `test_lora_update.py` 的源码与解释逐段对照。

---

## 1. 文件头部与导入（[行 1–46](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1-L46)）

```python
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ...
import json
import multiprocessing as mp
import unittest
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, List, Optional, Union

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.runners import SRTRunner
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    calculate_rouge_l,
    is_in_ci,
    popen_launch_server,
)
```

**解释**：
- `json`：Server 模式下把 adapter dict 序列化为命令行参数。
- `requests`：Server 模式下调用 HTTP API。
- `SRTRunner`：进程内 engine 模式运行器。
- `popen_launch_server` / `kill_process_tree`：启动/停止独立 server 子进程。
- `calculate_rouge_l`：对比动态与静态加载输出文本。
- `is_in_ci`：判断是否在 CI，决定跑全部还是仅 BASIC 用例。
- `register_cuda_ci` / `register_amd_ci`：CI 注册与耗时标注。

---

## 2. CI 注册与常量（[行 39–64](file:///d:/sglang/test/registered/lora/test_lora_update.py#L39-L64)）

```python
register_cuda_ci(
    est_time=487,
    stage="base-b",
    runner_config="1-gpu-large",
)
register_amd_ci(
    est_time=730,
    suite="stage-b-test-1-gpu-large-amd",
)

PROMPTS = [
    "SGL is a",
    "AI is a field of computer science focused on",
    "Computer science is the study of",
    "Write a short story.",
    "What are the main components of a computer?",
]

MEM_FRACTION_STATIC = 0.8
```

**解释**：
- 注册 CUDA/AMD CI 的预计运行时间和资源配置。
- `PROMPTS`：所有 forward 操作共享的 5 条 prompt。
- `MEM_FRACTION_STATIC`：静态显存分配比例。

---

## 3. 核心数据结构（[行 58–96](file:///d:/sglang/test/registered/lora/test_lora_update.py#L58-L96)）

```python
class OperationType(Enum):
    LOAD = "load"
    UNLOAD = "unload"
    FORWARD = "forward"
```

**解释**：定义三种可执行操作类型。

```python
@dataclass
class Operation:
    type: OperationType
    data: Optional[Any]
    expected_error: Optional[str] = None
    expected_implicit_evictions: Optional[set[str]] = None
```

**解释**：
- `type`：操作类型。
- `data`：操作数据。LOAD/UNLOAD 是 adapter 名或 dict；FORWARD 是 `(prompt, adapter)` 列表。
- `expected_error`：期望操作失败，并检查错误消息包含该子串。
- `expected_implicit_evictions`：由于 `max_loaded_loras` 限制，预期被隐式驱逐的 adapter 集合。

```python
@dataclass
class TestCase:
    description: str
    base: str
    max_loras_per_batch: int
    all_adapters: List[str]
    op_sequence: List[Operation]
    initial_adapters: Optional[List[str]] = None
    enable_lora: Optional[bool] = None
    max_lora_rank: Optional[int] = None
    lora_target_modules: Optional[List] = None
    max_new_tokens: int = 32
    max_loaded_loras: Optional[int] = None
```

**解释**：
- `base`：基础模型路径。
- `all_adapters`：该用例涉及的全部 adapter，用于“静态 pass”一次性加载。
- `op_sequence`：动态 load/unload/forward 操作序列。
- `initial_adapters`：启动时已加载的 adapter。

```python
def create_batch_data(adapters: Union[str, list]) -> List[tuple[str, str]]:
    if not isinstance(adapters, list):
        adapters = [adapters]
    return [(prompt, adapter) for prompt in PROMPTS for adapter in adapters]
```

**解释**：将 `PROMPTS` 与 adapter 列表做笛卡尔积，生成 `(prompt, lora_path)`。`adapter` 可为 `None`（使用 base 模型）。

---

## 4. BASIC_TESTS 用例 1：带初始 lora_paths 的动态更新（[行 98–209](file:///d:/sglang/test/registered/lora/test_lora_update.py#L98-L209)）

```python
TestCase(
    description="dynamic lora update with initial lora_paths",
    base="meta-llama/Llama-3.1-8B-Instruct",
    max_loras_per_batch=3,
    all_adapters=[
        "philschmid/code-llama-3-1-8b-text-to-sql-lora",
        "nvidia/llama-3.1-nemoguard-8b-topic-control",
        "pbevan11/llama-3.1-8b-ocr-correction",
    ],
    initial_adapters=[
        # Testing 3 supported lora-path formats.
        "philschmid/code-llama-3-1-8b-text-to-sql-lora",
        "nvidia/llama-3.1-nemoguard-8b-topic-control=nvidia/llama-3.1-nemoguard-8b-topic-control",
        {
            "lora_name": "pbevan11/llama-3.1-8b-ocr-correction",
            "lora_path": "pbevan11/llama-3.1-8b-ocr-correction",
            "pinned": False,
        },
    ],
```

**解释**：
- 启动时加载 3 个 adapter，并验证三种初始化格式：
  - 纯字符串
  - `name=path`
  - dict（带 `pinned`）

后续 `op_sequence` 关键操作：

```python
Operation(type=OperationType.LOAD, data="...", expected_error="already loaded"),
```

**解释**：重复加载已存在的 adapter，期望失败并提示 `already loaded`。

```python
Operation(type=OperationType.UNLOAD, data="..."),
Operation(type=OperationType.LOAD, data="..."),
```

**解释**：卸载后再重新加载。

```python
Operation(type=OperationType.FORWARD, data=create_batch_data([...3 adapters...])),
```

**解释**：对 3 个已加载 adapter 做 batch forward。

```python
Operation(type=OperationType.FORWARD, data=create_batch_data("nvidia/...")),
```

**解释**：对已卸载的 adapter 做 forward，期望失败（`never been loaded`）。

```python
Operation(type=OperationType.FORWARD, data=create_batch_data(None)),
```

**解释**：不使用任何 adapter，验证 base 模型也能正常推理。

---

## 5. BASIC_TESTS 用例 2：不带初始 lora_paths 的动态更新（[行 210–339](file:///d:/sglang/test/registered/lora/test_lora_update.py#L210-L339)）

```python
TestCase(
    description="dynamic lora update without initial lora_paths",
    base="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_lora_rank=256,
    lora_target_modules=["all"],
    max_loras_per_batch=4,
```

**解释**：启动时不加载任何 adapter，但显式启用 LoRA 并指定 rank、target modules。

```python
Operation(type=OperationType.FORWARD, data=create_batch_data([..., None])),
```

**解释**：forward batch 中混入 `None`，验证 LoRA adapter 与 base 模型混合 batch 推理。

---

## 6. TARGET_MODULE_TESTS（[行 340–456](file:///d:/sglang/test/registered/lora/test_lora_update.py#L340-L456)）

```python
TestCase(
    description="Test explicitly specified lora-target-modules.",
    ...
    lora_target_modules=["all"],
```

**解释**：显式指定 `lora_target_modules=["all"]`，可以加载 target modules 更多的 adapter。

```python
TestCase(
    description="Test inferred lora-target-modules - start with smaller adapter",
    ...
    initial_adapters=["algoprog/fact-generation-llama-3.1-8b-instruct-lora"],
    op_sequence=[
        ...
        Operation(type=OperationType.LOAD, data="nvidia/...", expected_error="incompatible"),
```

**解释**：从较小 adapter 隐式推断 `lora_target_modules` 后，再加载 target modules 更多的 adapter 会报 `incompatible`。

---

## 7. MAX_LORA_RANK_TESTS（[行 457–569](file:///d:/sglang/test/registered/lora/test_lora_update.py#L457-L569)）

```python
TestCase(
    description="Test explicitly specified max-lora-rank.",
    max_lora_rank=32,
    ...
    Operation(type=OperationType.LOAD, data="philschmid/...", expected_error="incompatible"),
```

**解释**：显式 `max_lora_rank=32`，rank=256 的 adapter 无法加载，报 `incompatible`。

```python
TestCase(
    description="test implicitly inferred max-lora-rank",
    initial_adapters=["pbevan11/llama-3.1-8b-ocr-correction"],  # r = 32
    op_sequence=[
        Operation(type=OperationType.LOAD, data="philschmid/...", expected_error="incompatible"),
```

**解释**：不指定 `max_lora_rank`，从初始 adapter 隐式推断为 32，同样无法加载 rank=256 的 adapter。

---

## 8. MAX_LOADED_LORAS_TESTS（[行 570–730](file:///d:/sglang/test/registered/lora/test_lora_update.py#L570-L730)）

```python
TestCase(
    description="Test max_loaded_loras limit as well as implicit eviction and reloading",
    max_loras_per_batch=2,
    max_loaded_loras=2,
```

**解释**：设置最多同时加载 2 个 adapter，测试 LRU 隐式驱逐。

```python
Operation(
    type=OperationType.LOAD,
    data="pbevan11/...",
    expected_implicit_evictions={"philschmid/..."},
),
```

**解释**：加载第 3 个 adapter 时，最早加载的非 pinned adapter 被隐式驱逐。

```python
Operation(
    type=OperationType.FORWARD,
    data=create_batch_data(["nvidia/...", "philschmid/..."]),
    expected_implicit_evictions={"pbevan11/..."},
),
```

**解释**：forward pass 需要的 adapter 不会被驱逐；不需要的 LRU adapter 被驱逐。

```python
TestCase(
    description="Test implicit eviction and reloading with pinned LoRA adapters",
    initial_adapters=[{
        "lora_name": "philschmid/...",
        "lora_path": "philschmid/...",
        "pinned": True,
    }],
```

**解释**：pinned adapter 不会被驱逐，可能导致非 pinned adapter 被驱逐或新加载报 `starvation`。

---

## 9. EVICTION_TESTS（[行 731–853](file:///d:/sglang/test/registered/lora/test_lora_update.py#L731-L853)）

```python
TestCase(
    description="dynamic lora update with evictions",
    all_adapters=[
        "lora1=philschmid/...",
        "lora2=nvidia/...",
        "lora3=pbevan11/...",
    ],
```

**解释**：使用自定义 `lora_name`（如 `lora1=...`）测试更复杂的驱逐与 pinned 状态切换。

```python
Operation(
    type=OperationType.LOAD,
    data={"lora_name": "lora2", "lora_path": "nvidia/...", "pinned": True},
    expected_error="starvation",
),
```

**解释**：已有 pinned adapter 占满槽位时，再加载 pinned adapter 会报 `starvation`。

---

## 10. ALL_TESTS 组合（[行 854–862](file:///d:/sglang/test/registered/lora/test_lora_update.py#L854-L862)）

```python
ALL_TESTS = (
    BASIC_TESTS
    + TARGET_MODULE_TESTS
    + MAX_LORA_RANK_TESTS
    + MAX_LOADED_LORAS_TESTS
    + EVICTION_TESTS
)

class LoRAUpdateTestSessionMode(Enum):
    ENGINE = "engine"
    SERVER = "server"
```

**解释**：汇总所有测试用例，并定义 engine/server 两种执行模式。

---

## 11. 会话基类 LoRAUpdateTestSessionBase（[行 868–951](file:///d:/sglang/test/registered/lora/test_lora_update.py#L868-L951)）

```python
class LoRAUpdateTestSessionBase:
    def __init__(self, *, testcase, model_path, lora_paths, ...):
        ...
        self.expected_adapters = set()
        if self.lora_paths:
            for adapter in self.lora_paths:
                if isinstance(adapter, dict):
                    lora_name = adapter["lora_name"]
                elif "=" in adapter:
                    lora_name = adapter.split("=")[0]
                else:
                    lora_name = adapter
                self.expected_adapters.add(lora_name)
```

**解释**：根据启动参数构建“期望已加载 adapter 集合”，后续每次 load/unload/forward 后维护该集合，用于和系统返回的 `loaded_adapters` 对比。

```python
def load_lora_adapter(self, ...): raise NotImplementedError
def unload_lora_adapter(self, ...): raise NotImplementedError
def forward(self, ...): raise NotImplementedError
```

**解释**：抽象方法，由 engine/server 子类实现。

---

## 12. Engine 会话实现 LoRAUpdateEngineTestSession（[行 952–1100](file:///d:/sglang/test/registered/lora/test_lora_update.py#L952-L1100)）

```python
class LoRAUpdateEngineTestSession(LoRAUpdateTestSessionBase):
    def __enter__(self):
        self.handle = SRTRunner(
            model_path=self.model_path,
            model_type="generation",
            lora_paths=self.lora_paths,
            max_lora_rank=self.max_lora_rank,
            ...
            disable_radix_cache=True,
        )
        self.handle.__enter__()
```

**解释**：进程内启动 SRTRunner engine。

```python
def load_lora_adapter(self, lora_name, lora_path=None, expected_error=None,
                      pinned=False, expected_implicit_evictions=None):
    response = self.handle.load_lora_adapter(
        lora_name=lora_name, lora_path=lora_path, pinned=pinned
    )
    if expected_error:
        self.testcase.assertFalse(response.success, ...)
        self.testcase.assertIn(expected_error, response.error_message, ...)
    else:
        self.expected_adapters.add(lora_name)
        if expected_implicit_evictions is not None:
            self.expected_adapters -= expected_implicit_evictions
        self.testcase.assertTrue(response.success, ...)
        self.testcase.assertEqual(set(response.loaded_adapters), self.expected_adapters, ...)
```

**解释**：调用 engine 的 load 接口，校验成功/失败及返回的已加载 adapter 集合。

```python
def unload_lora_adapter(self, lora_name):
    self.expected_adapters.remove(lora_name)
    response = self.handle.unload_lora_adapter(lora_name=lora_name)
    self.testcase.assertEqual(set(response.loaded_adapters), self.expected_adapters, ...)
```

**解释**：调用 engine 的 unload 接口，校验已加载 adapter 集合。

```python
def forward(self, prompts, lora_paths, max_new_tokens=32,
            expected_error=None, expected_implicit_evictions=None):
    try:
        response = self.handle.batch_forward(...)
    except ValueError as e:
        if expected_error:
            self.testcase.assertIn(expected_error, str(e))
            return str(e)
        raise e
    ...
    self.expected_adapters.update(
        [lora_path for lora_path in lora_paths if lora_path is not None]
    )
    if expected_implicit_evictions is not None:
        self.expected_adapters -= expected_implicit_evictions
    return response.output_strs
```

**解释**：调用 engine 的 batch_forward，校验输出长度，并维护 `expected_adapters`（forward 可能触发隐式加载/驱逐）。

---

## 13. Server 会话实现 LoRAUpdateServerTestSession（[行 1101–1282](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1101-L1282)）

```python
class LoRAUpdateServerTestSession(LoRAUpdateTestSessionBase):
    def __enter__(self):
        other_args = [
            "--cuda-graph-max-bs-decode", str(self.cuda_graph_max_bs_decode),
            "--max-loras-per-batch", str(self.max_loras_per_batch),
            "--lora-backend", self.lora_backend,
            "--random-seed", "42",
            "--max-running-request", "1",
            "--mem-fraction-static", str(MEM_FRACTION_STATIC),
            "--disable-radix-cache",
        ]
        if self.enable_lora:
            other_args.append("--enable-lora")
        if self.lora_paths:
            other_args.append("--lora-paths")
            for lora_path in self.lora_paths:
                if isinstance(lora_path, dict):
                    lora_path = json.dumps(lora_path)
                other_args.append(lora_path)
        ...
        self.handle = popen_launch_server(
            self.model_path,
            DEFAULT_URL_FOR_TEST,
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
```

**解释**：构造 sglang server 命令行参数并启动独立 server 子进程。

```python
def load_lora_adapter(self, ...):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/load_lora_adapter",
        json={"lora_name": lora_name, "lora_path": lora_path, "pinned": pinned},
    )
    if expected_error:
        self.testcase.assertEqual(response.status_code, 400, ...)
        self.testcase.assertIn(expected_error, response.text, ...)
    else:
        ...
        self.testcase.assertEqual(set(response.json()["loaded_adapters"]), self.expected_adapters, ...)
```

**解释**：通过 HTTP `/load_lora_adapter` 加载，校验状态码 400 或返回集合。

```python
def unload_lora_adapter(self, lora_name):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/unload_lora_adapter",
        json={"lora_name": lora_name},
    )
    ...
```

**解释**：通过 HTTP `/unload_lora_adapter` 卸载。

```python
def forward(self, prompts, lora_paths, max_new_tokens=32,
            expected_error=None, expected_implicit_evictions=None):
    response = requests.post(
        DEFAULT_URL_FOR_TEST + "/generate",
        json={
            "text": prompts,
            "lora_path": lora_paths,
            "sampling_params": {
                "temperature": 0,
                "top_k": 1,
                "max_new_tokens": max_new_tokens,
            },
        },
    )
    if expected_error:
        self.testcase.assertEqual(response.status_code, 400, ...)
    else:
        output = [r["text"] for r in response.json()]
        ...
```

**解释**：通过 HTTP `/generate` 执行推理，校验输出与隐式加载/驱逐。

---

## 14. 会话工厂（[行 1283–1295](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1283-L1295)）

```python
def LoRAUpdateTestSession(testcase, mode, **kwargs):
    if mode == LoRAUpdateTestSessionMode.ENGINE:
        return LoRAUpdateEngineTestSession(testcase=testcase, **kwargs)
    elif mode == LoRAUpdateTestSessionMode.SERVER:
        return LoRAUpdateServerTestSession(testcase=testcase, **kwargs)
    else:
        raise ValueError(f"Unrecognized mode: {mode!r}")
```

**解释**：根据模式返回 engine 或 server 会话实例。

---

## 15. 测试执行类 TestLoRADynamicUpdate（[行 1296–1500+](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1296-L1500)）

```python
class TestLoRADynamicUpdate(CustomTestCase):
    """
    This test case verifies that the SRT runner can dynamically load and unload LoRA adapters
    during a sequence of operations, and that the outputs of forward passes with dynamically loaded
    adapters match the outputs of forward passes with statically loaded adapters.
    """
```

**解释**：类核心目标：验证动态加载/卸载能力，以及动态与静态加载输出一致。

```python
def _run_operation_sequence(self, mode, base, initial_adapters, op_sequence,
                            max_loras_per_batch, ...):
    forward_outputs = []
    with LoRAUpdateTestSession(...) as session:
        for op in op_sequence:
            print("-" * 100)
            print(f"Running operation: {op.type} --- data: {op.data} --- mode: {mode} ---")
            if op_type == OperationType.LOAD:
                ...
            elif op_type == OperationType.UNLOAD:
                ...
            elif op_type == OperationType.FORWARD:
                prompts, adapters = zip(*data)
                result = session.forward(...)
                if not expected_error:
                    forward_outputs.append(result)
    return forward_outputs
```

**解释**：遍历操作序列，调用对应会话方法，收集成功 forward 的输出。

```python
def _run_dynamic_adapter_updates(self, mode, test_cases):
    for case_idx, test_case in enumerate(test_cases, start=1):
        # Dynamic pass
        dynamic_output = self._run_operation_sequence(
            mode=mode,
            initial_adapters=test_case.initial_adapters,
            ...
            op_sequence=test_case.op_sequence,
        )

        # Static pass
        forward_ops = [x for x in test_case.op_sequence
                       if x.type == OperationType.FORWARD and x.expected_error is None]
        static_output = self._run_operation_sequence(
            mode=mode,
            initial_adapters=test_case.all_adapters,
            ...
            op_sequence=forward_ops,
        )

        # Compare
        ROUGE_L_TOL = 0.9
        self.assertEqual(len(dynamic_output), len(static_output))
        for i, (dynamic, static) in enumerate(zip(dynamic_output, static_output), start=1):
            self.assertEqual(len(dynamic), len(static))
            for j, (d_out, s_out) in enumerate(zip(dynamic, static), start=1):
                rouge_score = calculate_rouge_l([d_out.strip()], [s_out.strip()])[0]
                self.assertGreaterEqual(rouge_score, ROUGE_L_TOL, ...)
```

**解释**：
- 对每个 `TestCase` 跑两遍：动态 load/unload 后再 forward；启动时静态加载所有 adapter 再 forward。
- 使用 ROUGE-L ≥ 0.9 判断两种方式的输出文本一致。

```python
def test_dynamic_lora_update_server(self):
    test_cases = BASIC_TESTS if is_in_ci() else ALL_TESTS
    self._run_dynamic_adapter_updates(
        mode=LoRAUpdateTestSessionMode.SERVER, test_cases=test_cases
    )
```

**解释**：当前实际执行的 server 模式测试。CI 环境下只跑 `BASIC_TESTS`，非 CI 跑 `ALL_TESTS`。

```python
# def test_dynamic_lora_update_engine(self):
#     ...
```

**解释**：engine 模式被注释掉，未启用。

---

## 16. /v1/models 接口测试（[行 1482–1547](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1482-L1547)）

```python
def test_v1_models_endpoint_with_lora(self):
    adapters = [
        "philschmid/code-llama-3-1-8b-text-to-sql-lora",
        "nvidia/llama-3.1-nemoguard-8b-topic-control",
    ]

    with LoRAUpdateTestSession(...) as session:
        response = requests.get(DEFAULT_URL_FOR_TEST + "/v1/models")
        self.assertEqual(len(models_data["data"]), 1)  # Only base model
        ...
        session.load_lora_adapter(lora_name="adapter1", lora_path=adapters[0])
        response = requests.get(...)
        self.assertEqual(len(models_data["data"]), 2)  # Base + 1 adapter
        ...
        session.unload_lora_adapter(lora_name="adapter1")
        self.assertEqual(len(models_data["data"]), 2)  # Base + 1 adapter
```

**解释**：验证 OpenAI 兼容 `/v1/models` 接口返回的模型列表随 adapter 加载/卸载动态变化。adapter 条目的 `parent` 指向 base model。

---

## 17. 主入口（[行 1550–1558](file:///d:/sglang/test/registered/lora/test_lora_update.py#L1550-L1558)）

```python
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
```

**解释**：设置多进程 `spawn` 启动方式，避免 CUDA fork 问题，然后运行 unittest。

以上是 `test_lora_update.py` 的源码与解释逐段对照。核心逻辑是：用声明式 `TestCase` + `Operation` 描述动态 LoRA 生命周期操作，分别通过 Engine/Server 会话执行，并用动态 pass 与静态 pass 的输出 ROUGE-L 对比来验证等价性。

---

