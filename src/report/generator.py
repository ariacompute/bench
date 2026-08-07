"""报告生成：汇总评测结果，生成 Markdown + JSON 报告"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import BenchmarkConfig, ScoringConfig

logger = logging.getLogger(__name__)


def _collect_results(results_dir: Path) -> dict[str, Any]:
    """收集 results/ 下所有结果数据"""
    data: dict[str, Any] = {"eval": {}, "perf": {}}

    eval_summary = results_dir / "eval_summary.json"
    if eval_summary.exists():
        with open(eval_summary, "r") as f:
            data["eval"] = json.load(f)

    perf_summary = results_dir / "perf_summary.json"
    if perf_summary.exists():
        with open(perf_summary, "r") as f:
            data["perf"] = json.load(f)

    return data


def _calculate_score(
    model_name: str,
    eval_data: dict[str, Any],
    perf_data: dict[str, Any],
    scoring: ScoringConfig,
) -> dict[str, float]:
    """计算综合评分"""
    # 能力分数（取所有量化配置中最佳值）
    best_cap = 0.0
    for quant, tasks in eval_data.items():
        scores: list[float] = []
        for task_name, result in tasks.items():
            if "accuracy" in result:
                scores.append(result["accuracy"])
            elif "exact_match" in result:
                scores.append(result["exact_match"])
            elif "strict_accuracy" in result:
                scores.append(result["strict_accuracy"])
            elif "pass_at_1" in result:
                scores.append(result["pass_at_1"])
        if scores:
            best_cap = max(best_cap, sum(scores) / len(scores))

    # 效率分数
    best_efficiency = 0.0
    for quant, result in perf_data.items():
        tps = 0.0
        speed_data = result.get("speed", {}).get("tokens_per_second", {})
        if speed_data:
            vals = [float(v) for v in speed_data.values() if isinstance(v, (int, float))]
            tps = sum(vals) / max(len(vals), 1) if vals else 0.0

        model_size = float(result.get("size", {}).get("model_size_mb", 1))
        peak_rss = float(result.get("memory", {}).get("peak_rss_mb", 1))
        denom = (model_size * peak_rss) ** 0.5
        efficiency = tps / denom * 100 if model_size > 0 and peak_rss > 0 else 0.0
        best_efficiency = max(best_efficiency, efficiency)

    # 长文本分数
    longcontext = 0.0
    for quant, tasks in eval_data.items():
        needle = tasks.get("needle_in_haystack", {}).get("accuracy", 0.0)
        longcontext = max(longcontext, needle)

    # 指令遵循分数
    instruction = 0.0
    for quant, tasks in eval_data.items():
        ifeval = tasks.get("ifeval", {}).get("strict_accuracy", 0.0)
        instruction = max(instruction, ifeval)

    total = (
        scoring.capability_weight * best_cap
        + scoring.efficiency_weight * best_efficiency
        + scoring.longcontext_weight * longcontext
        + scoring.instruction_weight * instruction
    )

    return {
        "capability": round(best_cap, 4),
        "efficiency": round(best_efficiency, 4),
        "longcontext": round(longcontext, 4),
        "instruction": round(instruction, 4),
        "total": round(total, 4),
    }


def _generate_markdown(data: dict[str, Any], models: list, scoring: ScoringConfig) -> str:
    """生成 Markdown 报告"""
    lines: list[str] = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines.append("# 端侧大模型综合评测报告\n")
    lines.append(f"**生成时间**: {now}\n")

    # 统计多引擎模型数
    evaled_names = list(data.get("eval", {}).keys()) or list(data.get("perf", {}).keys())
    lines.append("## 执行摘要\n")
    lines.append(f"- 评测模型数: {len(evaled_names)}")
    lines.append(
        "- 推理引擎: aria-engine (Aria 量化) | llama.cpp (GGUF) | transformers (HF 原生)"
    )
    lines.append("- 评测任务: MMLU, GSM8K, C-Eval, HumanEval, Needle-in-Haystack, IFEval")
    lines.append("- 性能指标: TTFT, tok/s, 峰值内存, 模型体积")
    lines.append("")

    # 排行榜
    lines.append("## 综合评分排行榜\n")
    scores: list[tuple[str, dict]] = []
    for model_name in sorted(evaled_names):
        model_eval = data.get("eval", {}).get(model_name, {})
        model_perf = data.get("perf", {}).get(model_name, {})
        score = _calculate_score(model_name, model_eval, model_perf, scoring)
        scores.append((model_name, score))
    scores.sort(key=lambda x: x[1]["total"], reverse=True)

    lines.append("| 排名 | 模型 | 引擎 | 能力 | 效率 | 长文本 | 指令遵循 | **综合分** |")
    lines.append("|------|------|------|------|------|--------|----------|------------|")

    # 从模型名推断引擎：aria 版本含 -aria，native 版本含 -native
    def _infer_engine(name: str) -> str:
        if name.endswith("-aria"):
            return "aria"
        elif name.endswith("-native"):
            prefix = name.rsplit("-native", 1)[0]
            parts = prefix.rsplit("-", 1)
            return parts[-1] if len(parts) > 1 else "native"
        return "-"

    for rank, (name, score) in enumerate(scores, 1):
        engine = _infer_engine(name)
        s = score
        lines.append(
            f"| {rank} | {name} | {engine} | {s['capability']:.4f} | "
            f"{s['efficiency']:.4f} | {s['longcontext']:.4f} | "
            f"{s['instruction']:.4f} | **{s['total']:.4f}** |"
        )
    lines.append("")

    # 详细结果
    lines.append("## 详细结果\n")
    for model_name in sorted(evaled_names):
        lines.append(f"### {model_name}\n")

        # 能力评测
        eval_data = data.get("eval", {}).get(model_name, {})
        if eval_data:
            lines.append("**能力评测**\n")
            lines.append("| 量化 | 任务 | 指标 | 值 |")
            lines.append("|------|------|------|----|")
            for quant, tasks in eval_data.items():
                for task_name, result in tasks.items():
                    for metric in ["accuracy", "exact_match", "pass_at_1", "strict_accuracy"]:
                        if metric in result:
                            lines.append(
                                f"| {quant} | {task_name} | {metric} | {result[metric]:.4f} |"
                            )
            lines.append("")

        # 性能评测
        perf_data = data.get("perf", {}).get(model_name, {})
        if perf_data:
            lines.append("**性能评测**\n")
            lines.append("| 量化 | TTFT-1024(ms) | tok/s-1024 | 峰值内存(MB) | 模型大小(MB) |")
            lines.append("|------|---------------|------------|-------------|-------------|")
            for quant, result in perf_data.items():
                speed = result.get("speed", {})
                ttft = speed.get("ttft_ms", {}).get("1024", "N/A")
                tps = speed.get("tokens_per_second", {}).get("1024", "N/A")
                rss = result.get("memory", {}).get("peak_rss_mb", "N/A")
                size = result.get("size", {}).get("model_size_mb", "N/A")
                lines.append(f"| {quant} | {ttft} | {tps} | {rss} | {size} |")
            lines.append("")

    return "\n".join(lines)


def generate_report(
    cfg: BenchmarkConfig | None,
    results_dir: Path,
    output_path: Path,
    fmt: str = "md",
) -> None:
    """生成评测报告

    Args:
        cfg: 评测配置（含评分权重、引擎映射）
        results_dir: 评测结果目录
        output_path: 输出文件路径
        fmt: 报告格式（md / json）
    """
    data = _collect_results(results_dir)

    if fmt == "json":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("JSON report saved to %s", output_path)
        return

    if fmt == "md":
        scoring = cfg.scoring if cfg else ScoringConfig()
        markdown = _generate_markdown(data, [], scoring)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        logger.info("Markdown report saved to %s", output_path)
        return

    logger.warning("unsupported format: %s", fmt)
