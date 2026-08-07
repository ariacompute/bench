"""CLI 主入口：aria-bench 命令行工具"""

import argparse
import logging
from pathlib import Path

from .config import load_config
from .env.download import download_model
from .env.engine import check_engine
from .env.hardware import detect_hardware
from .report.generator import generate_report

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aria Benchmark — 端侧模型综合评测")
    sub = parser.add_subparsers(dest="command")

    # ── env ──
    env_parser = sub.add_parser("env", help="准备推理环境：检查 + 下载模型")
    env_parser.add_argument("--models", nargs="*", help="指定模型名（默认全部）")
    env_parser.add_argument("--engines", nargs="*", help="仅检查指定引擎（默认全部）")
    env_parser.add_argument("--force-download", action="store_true", help="强制重新下载")
    env_parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    # ── hardware ──
    hw_parser = sub.add_parser("hardware", help="检测硬件环境")
    hw_parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    # ── eval ──
    eval_parser = sub.add_parser("eval", help="运行能力评测")
    eval_parser.add_argument("--models", nargs="*", help="指定模型名（默认全部）")
    eval_parser.add_argument("--tasks", nargs="*", help="指定任务（默认全部）")
    eval_parser.add_argument("--output", default="./results", help="输出目录")
    eval_parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    # ── perf ──
    perf_parser = sub.add_parser("perf", help="运行性能评测")
    perf_parser.add_argument("--models", nargs="*", help="指定模型名（默认全部）")
    perf_parser.add_argument("--hardware", default="local", help="硬件目标")
    perf_parser.add_argument("--output", default="./results", help="输出目录")
    perf_parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    # ── report ──
    report_parser = sub.add_parser("report", help="生成评测报告")
    report_parser.add_argument("--results", default="./results", help="结果目录")
    report_parser.add_argument("--output", default="./report.md", help="输出文件")
    report_parser.add_argument("--format", choices=["md", "json"], default="md", help="报告格式")
    report_parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    verbose = getattr(args, "verbose", False)
    setup_logging(verbose)

    if args.command == "env":
        cmd_env(args)
    elif args.command == "hardware":
        cmd_hardware(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "perf":
        cmd_perf(args)
    elif args.command == "report":
        cmd_report(args)


def cmd_env(args) -> None:
    """env 子命令：准备推理环境"""
    cfg = load_config()

    # 收集需要检查的引擎
    engine_set: set[str] = set()
    if args.engines:
        engine_set = set(args.engines)
    elif args.models:
        model_names = set(args.models)
        for m in cfg.models:
            if m.name in model_names:
                engine_set.add(m.engine)
    else:
        # 全部引擎
        engine_set = set(cfg.engines.keys())

    print(f"=== 引擎环境检测 ({len(engine_set)} engines) ===")
    all_ok = True
    for eng in sorted(engine_set):
        result = check_engine(eng)
        status_icon = "✓" if result.status.value == "ok" else "✗"
        print(f"  [{status_icon}] {eng}: {result.status.value}")
        if result.binary_path:
            print(f"       路径: {result.binary_path}")
        if result.version_string:
            print(f"       版本: {result.version_string}")
        if result.detail and result.status.value != "ok":
            print(f"       详情: {result.detail}")
        if result.status.value != "ok":
            all_ok = False
    print()

    # 下载模型
    selected_models = args.models if args.models else [m.name for m in cfg.models]
    print(f"=== 模型下载 ({len(selected_models)} models) ===")
    for model_name in selected_models:
        model = None
        for m in cfg.models:
            if m.name == model_name:
                model = m
                break
        if model is None:
            print(f"  [✗] {model_name}: 未在配置中找到")
            continue

        print(f"  [{model.engine}] {model_name} ← {model.source}")
        for quant in model.quantizations:
            state = download_model(
                model.engine, model.source, quantization=quant, force=args.force_download,
            )
            print(f"    [{state.value}] quant={quant}")

    print()
    if not all_ok:
        print("⚠ 部分引擎不可用，相关评测将被跳过。")
    else:
        print("✓ 所有引擎就绪。")


def cmd_hardware(args) -> None:
    """hardware 子命令：检测硬件环境"""
    import json

    hw = detect_hardware()
    print(json.dumps(hw, indent=2, ensure_ascii=False))


def cmd_eval(args) -> None:
    """eval 子命令：运行能力评测"""
    from .eval.runner import run_eval

    cfg = load_config()
    results_dir = Path(args.output)

    model_names = args.models if args.models else [m.name for m in cfg.models]

    # 获取全部任务
    all_tasks: list[str] = []
    if cfg.tasks.academic.mmlu:
        all_tasks.append("mmlu")
    if cfg.tasks.academic.gsm8k:
        all_tasks.append("gsm8k")
    if cfg.tasks.academic.ceval:
        all_tasks.append("ceval")
    if cfg.tasks.academic.humaneval:
        all_tasks.append("humaneval")
    if cfg.tasks.longcontext.needle_in_haystack:
        all_tasks.append("needle_in_haystack")
    if cfg.tasks.instruction.ifeval:
        all_tasks.append("ifeval")

    task_names = args.tasks if args.tasks else all_tasks

    print("=== 能力评测 ===")
    print(f"模型: {len(model_names)} 个")
    print(f"任务: {task_names}")
    print(f"输出: {results_dir}")
    print()

    run_eval(cfg, model_names, task_names, results_dir)
    print("\n✓ 能力评测完成")


def cmd_perf(args) -> None:
    """perf 子命令：运行性能评测"""
    from .perf.runner import run_perf

    cfg = load_config()
    results_dir = Path(args.output)

    model_names = args.models if args.models else [m.name for m in cfg.models]

    print("=== 性能评测 ===")
    print(f"模型: {len(model_names)} 个")
    print(f"硬件: {args.hardware}")
    print(f"输出: {results_dir}")
    print()

    run_perf(cfg, model_names, args.hardware, results_dir)
    print("\n✓ 性能评测完成")


def cmd_report(args) -> None:
    """report 子命令：生成评测报告"""
    cfg = load_config()
    results_dir = Path(args.results)
    output_path = Path(args.output)

    print("=== 生成报告 ===")
    print(f"结果目录: {results_dir}")
    print(f"输出文件: {output_path}")
    print(f"格式: {args.format}")
    print()

    generate_report(cfg, results_dir, output_path, fmt=args.format)
    print(f"✓ 报告已生成: {output_path}")


if __name__ == "__main__":
    main()
