# task.md — bench 实施任务清单

> 基于 [requirements.md](./requirements.md) 生成，按阶段分步实施。
> 每项任务完成后勾选，所有任务合入前须通过 `ruff check` + `pytest` 全绿。

---

## 多引擎适配重构（2026Q3）

### T22 — config.py 多引擎支持

- [x] `EngineType = Literal["aria", "llama_cpp", "transformers"]`
- [x] `ModelConfig` 增加 `engine` 字段、`compare_with` 字段
- [x] `EngineConfig` per-engine 配置（source/binary/timeout_seconds）
- [x] 配置加载时校验 `engine` 值合法、`compare_with` 引用存在
- [x] 单元测试更新

### T23 — benchmark_config.yaml 扩展

- [x] 现有 18 个模型添加 `engine: "aria"` 字段
- [x] 新增 18 个 repo 模型的原生引擎变体（`-native` 后缀 + `compare_with`）
- [x] 新增 4 个非仓库模型条目（MiniCPM/DeepSeek/Vlx-Seek/Step-Edge）
- [x] 新增 `engines` 配置段（aria / llama_cpp / transformers）

### T24 — env/engine.py 多引擎检测

- [x] `check_aria_engine()` — 检测 aria-engine binary
- [x] `check_llama_cpp()` — 检测 llama-cli binary
- [x] `check_transformers()` — 检测 transformers Python 包
- [x] `check_engine(engine_type)` → `EngineCheckResult`
- [x] `check_all_engines(engine_types)` 批量检测

### T25 — env/download.py 多格式下载

- [x] `download_aria()` — Aria quant bundle（HF snapshot_download）
- [x] `download_llama_cpp()` — GGUF 优先下载 `{repo}-GGUF`，降级 raw + 提示转换
- [x] `download_transformers()` — HF snapshot_download
- [x] `download_model(engine, source, quantization)` 调度器

### T26 — eval/engine_runner.py 适配器模式（核心）

- [x] `EngineAdapter` 抽象基类（run 方法签名）
- [x] `AriaEngineAdapter` — subprocess aria-engine run，解析 JSON lines
- [x] `LlamaCppAdapter` — subprocess llama-cli -m <gguf>，找 GGUF 文件
- [x] `TransformersAdapter` — subprocess python 临时脚本，HF pipeline 推理
- [x] `ADAPTER_REGISTRY` 注册表 + `run_inference()` 统一入口

### T27 — eval/perf runner 引擎信息传递

- [x] `eval/runner.py` — 传递 `model.engine` + `engine_cfg.timeout_seconds` 到各 task handler
- [x] `eval/academic.py` — 函数签名增加 `engine_type`/`timeout` 参数
- [x] `eval/instruction.py` — 同上
- [x] `eval/longcontext.py` — 同上
- [x] `perf/runner.py` — 传递 `model.engine` 到各 perf function
- [x] `perf/speed.py` — 增加 `engine_type`/`timeout` 参数
- [x] `perf/memory.py` — 同上
- [x] `perf/power.py` — 使用 model.engine 参数
- [x] `report/generator.py` — 排行榜增加引擎列

### T28 — CLI 多引擎适配

- [x] `cli env` — 检测多引擎状态、按引擎类型下载模型
- [x] `cli eval` — 自动传递 engine 信息
- [x] `cli perf` — 同上

### T29 — 文档更新

- [x] `AGENTS.md` — 架构图更新（三级适配器）、引擎映射表、常用命令更新
- [x] `requirements.md` — 引擎-模型映射表、多引擎配置结构、新验收标准
- [x] `task.md` — 本文件

### T30 — 单元测试更新

- [ ] `test_config.py` — 新增 engine 字段、EngineConfig、compare_with 校验、非法 engine 报错
- [ ] `test_env.py` — 新增 check_llama_cpp / check_transformers / check_engine 测试
- [ ] `test_eval.py` — 适配器模式测试（AriaEngineAdapter / LlamaCppAdapter / TransformersAdapter）
- [ ] `test_perf.py` — speed/memory mock 增加 engine_type 参数
- [ ] `test_report.py` — 报告增加引擎列显示
- [ ] `pytest tests/ -v` 全绿
- [ ] `ruff check src/ tests/` 零告警

---

## 已完成的 Phase 1-5 任务清单（参考）

### Phase 1: 项目骨架与配置

- [x] T1 — 项目骨架搭建 (pyproject.toml, src/, tests/, .gitignore, scripts/)
- [x] T2 — 配置文件与 schema 校验
- [x] T3 — CLI 框架
- [x] T4 — 模型下载模块
- [x] T5 — Aria Engine 部署
- [x] T6 — 硬件环境检测
- [x] T7 — env 子命令集成

### Phase 2: 能力评测

- [x] T8 — Aria Engine 推理适配层
- [x] T9 — 学术基准评测 (MMLU/GSM8K/C-Eval/HumanEval)
- [x] T10 — 长文本评测 (Needle-in-Haystack)
- [x] T11 — 指令遵循评测 (IFEval)
- [x] T12 — eval 子命令集成

### Phase 3: 性能评测

- [x] T13 — 推理速度测量
- [x] T14 — 内存占用测量
- [x] T15 — 模型体积测量
- [x] T16 — 功耗测量（Intel RAPL，可选）
- [x] T17 — perf 子命令集成

### Phase 4: 报告生成

- [x] T18 — 报告生成 (Markdown + JSON)

### Phase 5: 集成与验收

- [x] T19 — 一键脚本
- [x] T20 — 单元测试与代码质量
- [x] T21 — AGENTS.md 更新

---

## 实施顺序建议

```
T22 → T23              # 配置层（可并行）
  ↓
T24 + T25              # env 模块（可并行）
  ↓
T26                    # 适配器核心（被 T27/T28 依赖）
  ↓
T27 + T28              # runner 传递 + CLI 适配（可并行）
  ↓
T29 + T30              # 文档 + 测试（可并行）
```

---

> 下一步：T30 单元测试更新，并运行 pytest + ruff 验收。
