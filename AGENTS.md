# AGENTS.md — bench 工程上下文入口

> Agent 开发必读。本文件为渐进式披露入口，深入细节请阅读 requirements.md 及各子目录 README。

## 项目概述

aria bench：端侧大模型综合评测平台。对 model 仓库产出的 Aria quant bundle
及非仓库端侧模型进行能力+性能自动化评测，产出一对一的量化排名与选型建议。

## 评测范围

**17 个模型家族（9 repo + 4 非 repo），双引擎通道（aria + 原生）：**

| 来源 | 引擎 | 家族 |
|------|------|------|
| model 仓库 (aria 通道) | `aria` | Qwen / Gemma / LFM / Bonsai / Inkling / LingBot / Nanbeige / OpenPI / OpenVLA |
| model 仓库 (原生对比基线) | `llama_cpp` / `transformers` | 同上 9 家族，采用各自官方推荐引擎 |
| 非仓库端侧模型 | `llama_cpp` / `transformers` | MiniCPM / DeepSeek / Vlx-Seek / Step-Edge |

**引擎映射（官方推荐）：**

| 引擎 | 适用模型 |
|------|----------|
| `aria` | 全部 9 repo 家族（Aria quant bundle：q4/q3.26/q8 channel） |
| `llama_cpp` | Qwen / Gemma / Bonsai / Inkling / Nanbeige / MiniCPM / Step-Edge（GGUF） |
| `transformers` | LFM / LingBot / OpenPI / OpenVLA / DeepSeek / Vlx-Seek（HF 原生） |

## 架构

```
bench CLI (Python)
  ├── env setup (多引擎检测 + 模型下载)
  ├── eval (MMLU/GSM8K/C-Eval/HumanEval/Needle/IFEval)
  ├── perf (TTFT/tok-s/内存/体积/功耗)
  └── report (Markdown/JSON)
        ↓
  ┌──────────────────────────────────────────┐
  │ EngineAdapter (抽象接口)                   │
  ├──────────────────────────────────────────┤
  │ AriaEngineAdapter  │ llama-cli 子进程      │
  │ LlamaCppAdapter    │ llama-cli 子进程      │
  │ TransformersAdapter│ python 子进程         │
  └──────────────────────────────────────────┘
        ↓
  硬件层 (x86/ARM/CUDA/Metal)
```

## 目录结构

```
bench/
├── AGENTS.md              # 本文件
├── requirements.md         # 需求规格
├── task.md                 # 实施任务清单
├── pyproject.toml          # Python 项目元数据
├── benchmark_config.yaml   # 统一评测配置（模型/任务/硬件/引擎）
├── src/
│   ├── cli.py              # CLI 入口
│   ├── config.py           # 配置解析（EngineConfig + ModelConfig）
│   ├── env/                # 环境准备：多引擎检测、多格式模型下载
│   ├── eval/               # 能力评测：engine_runner 适配器模式 + 学术/长文/指令
│   ├── perf/               # 性能评测：速度/内存/体积/功耗
│   └── report/             # 报告生成
├── scripts/
│   └── run_benchmark.sh    # 一键评测脚本
├── tests/                  # 单元测试
└── results/                # 评测产出（Git ignored）
```

## 开发规范

1. **AGENTS.md 先行**：本文件约 100 行，渐进式披露
2. **requirements.md → 人工审核 → task.md**：先需求后实施
3. **验证通过**：`pytest` + `ruff check` 全绿，配置 YAML 校验通过

## 常用命令

```bash
# 环境准备（多引擎检测 + 模型下载）
python -m src.cli env --models all

# 硬件检测
python -m src.cli hardware

# 能力评测
python -m src.cli eval --models all --tasks all

# 性能评测
python -m src.cli perf --models all --hardware local

# 报告生成
python -m src.cli report --results results/ --output report.md

# 一键运行
bash scripts/run_benchmark.sh

# 代码校验
ruff check src/ tests/
pytest tests/ -v

# 配置校验
python -c "from src.config import load_config; load_config()"
```

## 进行中需求

- [x] requirements.md + task.md + Phase 1-5 实现
- [x] 多引擎适配架构（aria / llama_cpp / transformers）
- [x] repo 模型原生引擎对比基线
- [x] 非 repo 端侧模型接入（MiniCPM / DeepSeek / Vlx-Seek / Step-Edge）

## 注意事项

- 评测支持三种引擎：aria (Aria quant bundle)、llama_cpp (GGUF)、transformers (HF 原生)
- 每个 repo 模型产生两条评测记录：aria 通道 + 原生引擎对比基线
- VLA 模型 (OpenVLA / OpenPI / LingBot) 仅支持文本能力评测，不支持端侧推理对比
- 非 repo 模型 HF 源路径中 `vlx-seek` 和 `step-edge` 需人工确认
- 模型权重不提交 Git，通过 benchmark_config.yaml 配置路径，评测结果 `results/` 也不提交
