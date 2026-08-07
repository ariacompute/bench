# bench

[English](README.md) | [中文](README_cn.md)

对 **17 个模型家族**（9 repo + 4 非 repo）进行自动化能力+性能评测，支持 **三种推理引擎**（aria / llama_cpp / transformers），产出一对一的量化排名与选型建议。

## 快速开始

```bash
# 1. 环境检查（多引擎检测 + 硬件探测）
python -m src.cli env --models all

# 2. 模型下载（Aria bundle / GGUF / HuggingFace）
python -m src.cli env download --models all

# 3. 运行能力评测
python -m src.cli eval --models all --tasks all

# 4. 运行性能评测
python -m src.cli perf --models all --hardware local

# 5. 生成报告
python -m src.cli report --results results/ --output report.md

# 一键运行
bash scripts/run_benchmark.sh
```

## 评测范围

**17 个模型家族，双引擎通道（aria + 原生引擎对比基线）：**

| 来源 | 引擎 | 家族 |
|------|------|------|
| model 仓库（aria 通道） | `aria` | Qwen / Gemma / LFM / Bonsai / Inkling / LingBot / Nanbeige / OpenPI / OpenVLA |
| model 仓库（原生对比基线） | `llama_cpp` / `transformers` | 同上 9 家族，采用各自官方推荐引擎 |
| 非仓库端侧模型 | `llama_cpp` / `transformers` | MiniCPM / DeepSeek / Vlx-Seek / Step-Edge |

**引擎-模型映射（官方推荐）：**

| 引擎 | 适用模型 | 说明 |
|------|----------|------|
| `aria` | 全部 9 repo 家族 | Aria Hadamard+Lloyd-Max 量化（q4/q3.26/q8 channel） |
| `llama_cpp` | Qwen / Gemma / Bonsai / Inkling / Nanbeige / MiniCPM / Step-Edge | GGUF 格式，端侧主流方案 |
| `transformers` | LFM / LingBot / OpenPI / OpenVLA / DeepSeek / Vlx-Seek | HuggingFace 原生推理（VLA/LNN/MLA 架构） |

## 架构

```
bench CLI (Python)
  ├── env     (多引擎检测 + 多格式模型下载)
  ├── eval    (MMLU / GSM8K / C-Eval / HumanEval / Needle / IFEval)
  ├── perf    (TTFT / tokens/s / 内存 / 体积 / 功耗)
  └── report  (Markdown / JSON 排名报告)
        ↓
  ┌──────────────────────────────────────────┐
  │ EngineAdapter 抽象接口                     │
  ├──────────────────────────────────────────┤
  │ AriaEngineAdapter   → aria-engine 子进程   │
  │ LlamaCppAdapter     → llama-cli 子进程      │
  │ TransformersAdapter → Python 子进程          │
  └──────────────────────────────────────────┘
        ↓
  硬件层 (x86 / ARM / CUDA / Metal)
```

## 目录结构

```
bench/
├── AGENTS.md              # Agent 工程上下文入口与目录索引
├── requirements.md         # 需求规格（功能边界/异常/验收标准，人工审核制）
├── task.md                 # 实施任务清单
├── pyproject.toml          # Python 项目元数据
├── benchmark_config.yaml   # 统一评测配置（模型/任务/硬件/引擎/评分权重）
├── src/
│   ├── cli.py              # CLI 入口
│   ├── config.py           # 配置解析（EngineConfig + ModelConfig）
│   ├── env/                # 环境准备：多引擎检测 + 多格式模型下载
│   ├── eval/               # 能力评测：engine_runner 适配器 + 学术/长文/指令
│   ├── perf/               # 性能评测：速度/内存/体积/功耗
│   └── report/             # 报告生成
├── scripts/
│   └── run_benchmark.sh    # 一键评测脚本
├── tests/                  # 单元测试（69 用例，ruff + pytest 全绿）
└── results/                # 评测产出（Git ignored）
```

## 常用命令

```bash
# 环境检测
python -m src.cli env --models all

# 硬件信息
python -m src.cli hardware

# 单项评测示例
python -m src.cli eval --models qwen3-0.6b-aria --tasks mmlu
python -m src.cli perf --models gemma3-1b-native --hardware local

# 代码校验
ruff check src/ tests/
pytest tests/ -v

# 配置校验
python -c "from src.config import load_config; load_config()"
```

## 工程规范

本仓库遵循 Harness Engineering 理念：

- [`AGENTS.md`](AGENTS.md)：Agent 工程上下文入口与目录索引
- [`requirements.md`](requirements.md)：需求规格（功能边界/异常/验收标准，人工审核制）
- [`task.md`](task.md)：实施任务清单

## 许可

MIT License. 详见 [LICENSE](./LICENSE).
