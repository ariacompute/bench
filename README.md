# bench

[English](README.md) | [中文](README_cn.md)

Automated capability and performance evaluation for **17 model families** (9 repo + 4 non-repo) across **three inference engines** (aria / llama_cpp / transformers), producing one-to-one quantized rankings and deployment recommendations.

## Quick Start

```bash
# 1. Environment check (multi-engine detection + hardware probe)
python -m src.cli env --models all

# 2. Model download (Aria bundle / GGUF / HuggingFace)
python -m src.cli env download --models all

# 3. Run capability evaluation
python -m src.cli eval --models all --tasks all

# 4. Run performance evaluation
python -m src.cli perf --models all --hardware local

# 5. Generate report
python -m src.cli report --results results/ --output report.md

# One-click run
bash scripts/run_benchmark.sh
```

## Evaluation Scope

**17 model families, dual engine channels (aria + native baseline):**

| Source | Engine | Families |
|--------|--------|----------|
| model repo (aria channel) | `aria` | Qwen / Gemma / LFM / Bonsai / Inkling / LingBot / Nanbeige / OpenPI / OpenVLA |
| model repo (native baseline) | `llama_cpp` / `transformers` | Same 9 families, each using its officially recommended engine |
| non-repo edge models | `llama_cpp` / `transformers` | MiniCPM / DeepSeek / Vlx-Seek / Step-Edge |

**Engine-model mapping (official recommendations):**

| Engine | Supported models | Notes |
|--------|------------------|-------|
| `aria` | All 9 repo families | Aria Hadamard+Lloyd-Max quantization (q4/q3.26/q8) |
| `llama_cpp` | Qwen / Gemma / Bonsai / Inkling / Nanbeige / MiniCPM / Step-Edge | GGUF format, mainstream edge deployment |
| `transformers` | LFM / LingBot / OpenPI / OpenVLA / DeepSeek / Vlx-Seek | HuggingFace native inference (VLA/LNN/MLA architectures) |

## Architecture

```
bench CLI (Python)
  ├── env     (multi-engine detection + multi-format model download)
  ├── eval    (MMLU / GSM8K / C-Eval / HumanEval / Needle / IFEval)
  ├── perf    (TTFT / tokens/s / memory / size / power)
  └── report  (Markdown / JSON ranking report)
        ↓
  ┌──────────────────────────────────────────┐
  │ EngineAdapter (abstract interface)        │
  ├──────────────────────────────────────────┤
  │ AriaEngineAdapter   → aria-engine subprocess  │
  │ LlamaCppAdapter     → llama-cli subprocess     │
  │ TransformersAdapter → Python subprocess        │
  └──────────────────────────────────────────┘
        ↓
  Hardware layer (x86 / ARM / CUDA / Metal)
```

## Directory Structure

```
bench/
├── AGENTS.md              # Agent engineering context entry & directory index
├── requirements.md         # Requirements spec (scope/exceptions/acceptance, human-reviewed)
├── task.md                 # Implementation task checklist
├── pyproject.toml          # Python project metadata
├── benchmark_config.yaml   # Unified benchmark config (models/tasks/hardware/engines/scoring)
├── src/
│   ├── cli.py              # CLI entry point
│   ├── config.py           # Config parsing (EngineConfig + ModelConfig)
│   ├── env/                # Environment setup: multi-engine detection + multi-format download
│   ├── eval/               # Capability eval: engine_runner adapters + academic/longcontext/instruction
│   ├── perf/               # Performance eval: speed/memory/size/power
│   └── report/             # Report generation
├── scripts/
│   └── run_benchmark.sh    # One-click benchmark script
├── tests/                  # Unit tests (69 cases, ruff + pytest all green)
└── results/                # Benchmark outputs (Git ignored)
```

## Common Commands

```bash
# Environment detection
python -m src.cli env --models all

# Hardware info
python -m src.cli hardware

# Single-task evaluation examples
python -m src.cli eval --models qwen3-0.6b-aria --tasks mmlu
python -m src.cli perf --models gemma3-1b-native --hardware local

# Code quality checks
ruff check src/ tests/
pytest tests/ -v

# Config validation
python -c "from src.config import load_config; load_config()"
```

## Engineering Standards

This repository follows Harness Engineering principles:

- [`AGENTS.md`](AGENTS.md): Agent engineering context entry & directory index
- [`requirements.md`](requirements.md): Requirements spec (scope/exceptions/acceptance, human-reviewed)
- [`task.md`](task.md): Implementation task checklist

## License

MIT License. See [LICENSE](./LICENSE) for details.
