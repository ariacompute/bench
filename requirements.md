# requirements.md — bench 需求规格

> **状态**：已审核，可生成 task.md  
> **关联**：端侧大模型综合评测方案  
> **前置依赖**：model 仓库（aria quant bundle 产出）、Aria Engine、llama.cpp、HuggingFace Transformers

---

## 1. 功能边界

### 1.1 模型覆盖范围（双引擎通道）

| 通道 | 家族数 | 引擎 | 说明 |
|------|--------|------|------|
| model 仓库 — aria | 9 | `aria` | Aria Hadamard+Lloyd-Max 量化（q4/q3.26/q8 channel） |
| model 仓库 — 原生基线 | 9 | `llama_cpp` / `transformers` | 各家族官方推荐引擎（见 1.2） |
| 非仓库端侧模型 | 4 | `llama_cpp` / `transformers` | MiniCPM / DeepSeek / Vlx-Seek / Step-Edge |

### 1.2 引擎-模型映射

| 模型家族 | aria 引擎 | 原生引擎 | 原生引擎选型说明 |
|----------|-----------|----------|-----------------|
| Qwen | ✓ | `llama_cpp` | GGUF，阿里官方推荐端侧方案 |
| Gemma | ✓ | `llama_cpp` | GGUF，Google 官方端侧支持 |
| LFM | ✓ | `transformers` | Liquid Neural Network，HF 原生支持 |
| Bonsai | ✓ | `llama_cpp` | 标准 Transformer LLM，GGUF 量化 |
| Inkling | ✓ | `llama_cpp` | 标准小型 LLM，GGUF 量化 |
| LingBot | ✓ | `transformers` | VLA 模型，需 HF 特定架构 |
| Nanbeige | ✓ | `llama_cpp` | 标准中文 LLM，GGUF 量化 |
| OpenPI | ✓ | `transformers` | VLA 模型，LeRobot pipeline |
| OpenVLA | ✓ | `transformers` | VLA 模型，Prismatic 架构 |
| MiniCPM | — | `llama_cpp` | OpenBMB 官方 GGUF 支持 |
| DeepSeek | — | `transformers` | MLA 注意力，HF 原生适配 |
| Vlx-Seek | — | `transformers` | 多模态 VLM |
| Step-Edge | — | `llama_cpp` | StepFun 端侧 GGUF |

### 1.3 多引擎适配架构

```
EngineDispatcher
├── AriaEngineAdapter  → subprocess: aria-engine run
├── LlamaCppAdapter    → subprocess: llama-cli -m <gguf> -p <prompt>
└── TransformersAdapter → subprocess: python -c (transformers pipeline)
```

**统一接口**: `run_inference(model_path, engine_type, prompt, max_tokens, temperature, timeout, **kwargs) → dict`

**返回值统一**: `{"text": str, "tokens": int, "finish_reason": str, "ttft_ms": float|None, "tokens_per_second": float|None, "error": str|None}`

### 1.4 量化格式差异

| 引擎 | 量化格式 | 来源 |
|------|----------|------|
| `aria` | q4_channel / q3.26_channel / q8_channel | `ariacompute/*` HF |
| `llama_cpp` | q4_k_m / q8_0（GGUF） | 优先 `{repo}-GGUF`，降级原始 safetensors + 提示转换 |
| `transformers` | fp16 / int8（HF 原生） | 原始 HF repo |

---

## 2. 配置结构变更

### 2.1 模型条目新增字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `engine` | `str` | 推理引擎：`aria` / `llama_cpp` / `transformers`（必填） |
| `compare_with` | `str?` | 关联对比模型名（aria↔native 互指，可选） |

### 2.2 引擎配置段（新增）

```yaml
engines:
  aria:
    source: "https://github.com/ariacompute/engine"
    binary: "aria-engine"
    timeout_seconds: 3600
  llama_cpp:
    source: "https://github.com/ggerganov/llama.cpp"
    binary: "llama-cli"
    timeout_seconds: 3600
  transformers:
    source: "https://github.com/huggingface/transformers"
    timeout_seconds: 3600
```

---

## 3. 异常处理补充

| 异常场景 | 处理策略 |
|----------|----------|
| 引擎二进制不可用（llama-cli） | WARNING + 跳过该引擎的模型 |
| transformers 未安装 | WARNING + 跳过 transformers 引擎的模型 |
| GGUF 文件未找到 | ERROR + 提示使用 convert_hf_to_gguf.py 转换 |
| VLA 模型评测（transformers） | 文本任务正常执行，动作生成由引擎适配器处理 |

---

## 4. 验收标准（新增）

| # | 验收项 | 标准 |
|---|--------|------|
| AC9 | 多引擎检测 | `cli env` 检测三种引擎状态（aria / llama_cpp / transformers） |
| AC10 | aria-native 对比 | repo 模型产生 aria + 原生两条评测记录 |
| AC11 | 非 repo 模型评测 | 4 个非仓库模型按各自引擎通过评测 |
| AC12 | 报告展示引擎列 | 排行榜包含引擎类型列 |

---

## 5. 已确认决策（更新）

| # | 决策项 | 方案 |
|---|--------|------|
| D7 | repo 模型原生引擎对比 | 同一模型产生 aria + native 两条评测记录，报告并列对比 |
| D8 | 引擎分配 | Qwen/Gemma/Bonsai/Inkling/Nanbeige/Step-Edge → llama_cpp；LFM/LingBot/OpenPI/OpenVLA/DeepSeek/Vlx-Seek → transformers |
