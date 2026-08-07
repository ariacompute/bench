"""perf 性能评测编排器"""

import json
import logging
from pathlib import Path
from typing import Any

from ..config import BenchmarkConfig, ModelConfig

logger = logging.getLogger(__name__)


def _get_bundle_path(model: ModelConfig, quant: str, results_dir: Path) -> str:
    return str(results_dir / "bundles" / model.name / quant)


def _find_model(cfg: BenchmarkConfig, name: str) -> ModelConfig | None:
    for m in cfg.models:
        if m.name == name:
            return m
    return None


def run_perf(
    cfg: BenchmarkConfig,
    model_names: list[str],
    hardware_target: str,
    results_dir: Path,
) -> None:
    """运行性能评测主流程"""
    from .memory import measure_memory
    from .power import measure_power
    from .size import measure_size
    from .speed import measure_speed

    perf_results: dict[str, dict] = {}

    for model_name in model_names:
        model = _find_model(cfg, model_name)
        if model is None:
            logger.warning("model '%s' not found in config, skipping", model_name)
            continue

        logger.info("=== %s [engine=%s] ===", model.name, model.engine)

        # 获取引擎超时
        engine_cfg = cfg.engines.get(model.engine)
        timeout = engine_cfg.timeout_seconds if engine_cfg else 600

        perf_results[model_name] = {}

        for quant in model.quantizations:
            bundle_path = _get_bundle_path(model, quant, results_dir)
            output_dir = results_dir / model.name / quant
            output_dir.mkdir(parents=True, exist_ok=True)
            result_path = output_dir / "perf.json"

            # 断点续跑
            if result_path.exists():
                logger.info("[%s/%s] existing perf results found, skipping", model_name, quant)
                with open(result_path, "r") as f:
                    perf_results[model_name][quant] = json.load(f)
                continue

            logger.info("[%s/%s] measuring speed...", model_name, quant)
            speed = measure_speed(model, quant, bundle_path, output_dir, model.engine, timeout)

            logger.info("[%s/%s] measuring memory...", model_name, quant)
            memory = measure_memory(model, quant, bundle_path, output_dir, model.engine, timeout)

            logger.info("[%s/%s] measuring size...", model_name, quant)
            size = measure_size(model, quant, bundle_path, output_dir)

            logger.info("[%s/%s] measuring power...", model_name, quant)
            power = measure_power(model, quant, bundle_path, output_dir)

            result: dict[str, Any] = {
                "model": model.name,
                "engine": model.engine,
                "quantization": quant,
                "hardware": hardware_target,
                "speed": speed,
                "memory": memory,
                "size": size,
            }
            if power["available"]:
                result["power"] = {
                    "energy_joules": power.get("energy_joules"),
                    "power_watts": power.get("power_watts"),
                }

            perf_results[model_name][quant] = result

            with open(result_path, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

            logger.info(
                "[%s/%s] perf done: ttft=%s, tps=%s, rss=%s, size=%s",
                model_name,
                quant,
                speed.get("ttft_ms", {}).get("1024", "N/A"),
                speed.get("tokens_per_second", {}).get("1024", "N/A"),
                memory.get("peak_rss_mb", "N/A"),
                size.get("model_size_mb", "N/A"),
            )

    # 保存汇总
    summary_path = results_dir / "perf_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(perf_results, f, indent=2, ensure_ascii=False)
    logger.info("perf summary saved to %s", summary_path)
