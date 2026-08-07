"""eval 评测编排器"""

import json
import logging
from pathlib import Path
from typing import Any

from ..config import BenchmarkConfig, ModelConfig

logger = logging.getLogger(__name__)


def _get_bundle_path(model: ModelConfig, quant: str, results_dir: Path) -> str:
    """获取 bundle 路径（模型下载目录）"""
    return str(results_dir / "bundles" / model.name / quant)


def _find_model(cfg: BenchmarkConfig, name: str) -> ModelConfig | None:
    """在配置中查找模型。"""
    for m in cfg.models:
        if m.name == name:
            return m
    return None


def run_eval(
    cfg: BenchmarkConfig,
    model_names: list[str],
    task_names: list[str],
    results_dir: Path,
) -> None:
    """运行能力评测主流程

    遍历 模型 × 量化配置 × 任务 三重循环，按引擎类型分发推理。
    """
    from .academic import run_ceval, run_gsm8k, run_humaneval, run_mmlu
    from .instruction import run_ifeval
    from .longcontext import run_needle_in_haystack

    # 任务注册表
    task_handlers: dict[str, Any] = {
        "mmlu": run_mmlu,
        "gsm8k": run_gsm8k,
        "ceval": run_ceval,
        "humaneval": run_humaneval,
        "needle_in_haystack": run_needle_in_haystack,
        "ifeval": run_ifeval,
    }

    all_results: dict[str, dict] = {}

    for model_name in model_names:
        model = _find_model(cfg, model_name)
        if model is None:
            logger.warning("model '%s' not found in config, skipping", model_name)
            continue

        logger.info("=== %s [engine=%s] ===", model.name, model.engine)

        # 获取引擎配置（含超时）
        engine_cfg = cfg.engines.get(model.engine)
        timeout = engine_cfg.timeout_seconds if engine_cfg else 600

        for quant in model.quantizations:
            bundle_path = _get_bundle_path(model, quant, results_dir)
            output_dir = results_dir / model.name / quant
            output_dir.mkdir(parents=True, exist_ok=True)

            for task_name in task_names:
                if task_name not in task_handlers:
                    logger.warning("unknown task: %s, skipping", task_name)
                    continue

                logger.info("[%s/%s/%s] running...", model_name, quant, task_name)

                handler = task_handlers[task_name]

                # needle_in_haystack 需要额外参数
                if task_name == "needle_in_haystack":
                    nh_config = cfg.tasks.longcontext
                    result = handler(
                        model, quant, bundle_path, output_dir,
                        model.engine, timeout,
                        context_lengths=nh_config.context_lengths,
                        depth_percent=nh_config.depth_percent,
                    )
                else:
                    result = handler(
                        model, quant, bundle_path, output_dir, model.engine, timeout,
                    )

                # 保存单任务摘要
                all_results.setdefault(model_name, {}).setdefault(quant, {})[task_name] = {
                    k: v for k, v in result.items() if k != "results"
                }

                logger.info(
                    "[%s/%s/%s] done: %s",
                    model_name,
                    quant,
                    task_name,
                    {k: v for k, v in result.items() if k != "results"},
                )

    # 保存汇总
    summary_path = results_dir / "eval_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("eval summary saved to %s", summary_path)
