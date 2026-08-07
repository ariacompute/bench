"""Benchmark 配置加载与校验模块。

从 benchmark_config.yaml 读取模型、任务、硬件、指标与引擎配置，
执行字段级校验，返回结构化 Config 对象。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import yaml

# ── 引擎类型 ──
ENGINE_TYPES = ("aria", "llama_cpp", "transformers")
EngineType = Literal["aria", "llama_cpp", "transformers"]


def _parse_engine_type(raw: str) -> EngineType:
    v = raw.strip().lower()
    if v not in ENGINE_TYPES:
        raise ConfigError(f"未知引擎类型 '{raw}'，支持: {ENGINE_TYPES}")
    return v  # type: ignore[return-value]


# ── 错误类型 ──
class ConfigError(Exception):
    """配置校验错误。"""


# ── 配置数据类 ──
@dataclass
class EngineConfig:
    """单个推理引擎的安装/运行配置。"""

    engine_type: EngineType
    source: str  # GitHub / PyPI 地址
    release: str = "latest"
    binary: Optional[str] = None  # 子进程可执行文件名（aria / llama_cpp 使用）
    version: str = ""
    timeout_seconds: int = 3600
    description: str = ""


@dataclass
class ModelConfig:
    """单个被测模型的配置。"""

    name: str  # 唯一标识，如 "qwen3-0.6b-aria"
    family: str  # 模型家族，如 "qwen"
    source: str  # HF repo_id 或 ariacompute 前缀
    engine: EngineType
    quantizations: List[str] = field(default_factory=list)
    max_context: int = 4096
    compare_with: Optional[str] = None  # 关联的对比模型名（如 aria ↔ native）

    @property
    def is_native(self) -> bool:
        """是否为原生引擎评测通道。"""
        return self.engine != "aria"

    @property
    def is_vla(self) -> bool:
        """VLA（Vision-Language-Action）模型不适用于纯文本评测。"""
        return self.family in ("openvla", "openpi", "lingbot")


@dataclass
class AcademicTasks:
    mmlu: bool = True
    gsm8k: bool = True
    ceval: bool = True
    humaneval: bool = True


@dataclass
class LongContextConfig:
    needle_in_haystack: bool = True
    context_lengths: List[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768])
    depth_percent: List[int] = field(default_factory=lambda: [0, 25, 50, 75, 100])


@dataclass
class InstructionTasks:
    ifeval: bool = True


@dataclass
class TaskConfig:
    academic: AcademicTasks = field(default_factory=AcademicTasks)
    longcontext: LongContextConfig = field(default_factory=LongContextConfig)
    instruction: InstructionTasks = field(default_factory=InstructionTasks)


@dataclass
class HardwareTarget:
    name: str
    type: str  # x86-cpu, aarch64, android, webgpu ...
    desc: str = ""


@dataclass
class HardwareConfig:
    targets: List[HardwareTarget] = field(default_factory=list)


@dataclass
class MetricConfig:
    capability: List[str] = field(default_factory=list)
    performance: List[str] = field(default_factory=list)


@dataclass
class ScoringConfig:
    capability_weight: float = 0.40
    efficiency_weight: float = 0.25
    longcontext_weight: float = 0.10
    instruction_weight: float = 0.05
    multimodal_weight: float = 0.15
    deploy_weight: float = 0.05


@dataclass
class BenchmarkConfig:
    name: str = ""
    version: str = ""
    output_dir: str = "./results"
    models: List[ModelConfig] = field(default_factory=list)
    engines: Dict[str, EngineConfig] = field(default_factory=dict)
    tasks: TaskConfig = field(default_factory=TaskConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


# ── 配置加载与校验 ──
def load_config(path: Optional[str] = None) -> BenchmarkConfig:
    """加载并校验 benchmark_config.yaml。

    Args:
        path: YAML 文件路径，None 则回退环境变量 BENCHMARK_CONFIG
              或仓库根目录 benchmark_config.yaml。

    Returns:
        校验后的 BenchmarkConfig。

    Raises:
        ConfigError: 配置不合法。
    """
    if path is None:
        path = os.environ.get(
            "BENCHMARK_CONFIG",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "benchmark_config.yaml"),
        )
        path = os.path.normpath(path)

    if not os.path.isfile(path):
        raise ConfigError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ConfigError("配置文件为空")

    # 顶层结构
    bench = raw.get("benchmark", raw)
    model_list = raw.get("models", [])
    task_raw = raw.get("tasks", {})
    hw_raw = raw.get("hardware", {})
    metric_raw = raw.get("metrics", {})
    engine_raw = raw.get("engines", {})
    scoring_raw = raw.get("scoring", {})

    # 解析模型
    models: List[ModelConfig] = []
    model_names: set = set()
    for m in model_list:
        name = m.get("name")
        if not name:
            raise ConfigError("模型配置缺少 name 字段")
        if name in model_names:
            raise ConfigError(f"重复的模型名: {name}")
        model_names.add(name)

        engine_raw_val = m.get("engine", "aria")
        if not isinstance(engine_raw_val, str):
            raise ConfigError(f"模型 '{name}' engine 字段必须为字符串")
        engine = _parse_engine_type(engine_raw_val)

        mc = ModelConfig(
            name=name,
            family=m.get("family", name),
            source=m["source"],
            engine=engine,
            quantizations=m.get("quantizations", []),
            max_context=m.get("max_context", 4096),
            compare_with=m.get("compare_with"),
        )
        models.append(mc)

    # 解析引擎
    engines: Dict[str, EngineConfig] = {}
    for eng_name, eng_data in engine_raw.items():
        engines[eng_name] = EngineConfig(
            engine_type=_parse_engine_type(eng_name),
            source=eng_data.get("source", ""),
            release=eng_data.get("release", "latest"),
            binary=eng_data.get("binary"),
            version=eng_data.get("version", ""),
            timeout_seconds=eng_data.get("timeout_seconds", 3600),
            description=eng_data.get("description", ""),
        )

    # 校验模型中引用的引擎在 engines 段存在
    for mc in models:
        if mc.engine not in engines:
            raise ConfigError(
                f"模型 '{mc.name}' 引用了未在 engines 段定义的引擎 '{mc.engine}'"
            )

    # 校验 compare_with 引用合法
    for mc in models:
        if mc.compare_with:
            if mc.compare_with not in model_names:
                raise ConfigError(
                    f"模型 '{mc.name}' compare_with='{mc.compare_with}' 未找到"
                )

    # 解析任务
    academic_raw = task_raw.get("academic", [])
    lc_raw = task_raw.get("longcontext", [])
    instr_raw = task_raw.get("instruction", [])

    academic = AcademicTasks(
        mmlu="mmlu" in academic_raw,
        gsm8k="gsm8k" in academic_raw,
        ceval="ceval" in academic_raw,
        humaneval="humaneval" in academic_raw,
    )

    lc = LongContextConfig()
    if lc_raw:
        niah = lc_raw[0] if isinstance(lc_raw, list) else lc_raw
        if isinstance(niah, dict) and "needle_in_haystack" in niah:
            lc.needle_in_haystack = True
            lc.context_lengths = niah["needle_in_haystack"].get(
                "context_lengths", [4096, 8192, 16384, 32768],
            )
            lc.depth_percent = niah["needle_in_haystack"].get(
                "depth_percent", [0, 25, 50, 75, 100],
            )

    instr = InstructionTasks()
    if instr_raw:
        instr.ifeval = "ifeval" in instr_raw

    tasks = TaskConfig(academic=academic, longcontext=lc, instruction=instr)

    # 解析硬件
    hw_targets = []
    for t in hw_raw.get("targets", []):
        hw_targets.append(HardwareTarget(
            name=t.get("name", ""),
            type=t.get("type", "x86-cpu"),
            desc=t.get("desc", ""),
        ))
    hardware = HardwareConfig(targets=hw_targets)

    # 解析指标
    metrics = MetricConfig(
        capability=metric_raw.get("capability", []),
        performance=metric_raw.get("performance", []),
    )

    # 解析评分权重
    scoring = ScoringConfig(
        capability_weight=scoring_raw.get("capability_weight", 0.40),
        efficiency_weight=scoring_raw.get("efficiency_weight", 0.25),
        longcontext_weight=scoring_raw.get("longcontext_weight", 0.10),
        instruction_weight=scoring_raw.get("instruction_weight", 0.05),
        multimodal_weight=scoring_raw.get("multimodal_weight", 0.15),
        deploy_weight=scoring_raw.get("deploy_weight", 0.05),
    )

    return BenchmarkConfig(
        name=bench.get("name", ""),
        version=bench.get("version", ""),
        output_dir=bench.get("output_dir", "./results"),
        models=models,
        engines=engines,
        tasks=tasks,
        hardware=hardware,
        metrics=metrics,
        scoring=scoring,
    )


def get_engine_for_model(
    model: ModelConfig, engines: Dict[str, EngineConfig]
) -> EngineConfig:
    """获取模型对应的引擎配置。"""
    return engines[model.engine]
