#!/usr/bin/env bash
# run_benchmark.sh — 一键评测脚本
# 用法: bash scripts/run_benchmark.sh [--skip-env] [--skip-eval] [--skip-perf]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${BENCH_DIR}/benchmark_config.yaml"
PYTHON="${PYTHON:-python3}"

SKIP_ENV=false
SKIP_EVAL=false
SKIP_PERF=false

# 解析参数
for arg in "$@"; do
    case "$arg" in
        --skip-env)  SKIP_ENV=true ;;
        --skip-eval) SKIP_EVAL=true ;;
        --skip-perf) SKIP_PERF=true ;;
        *) echo "未知参数: $arg"; exit 1 ;;
    esac
done

echo "========================================="
echo "  aria bench 一键评测"
echo "========================================="

cd "$BENCH_DIR"

# Phase 1: 环境准备
if [ "$SKIP_ENV" = false ]; then
    echo -e "\n>>> Phase 1: 环境准备 <<<"
    $PYTHON -m src.cli env setup --config "$CONFIG" || {
        echo "ERROR: 环境准备失败"
        exit 1
    }
else
    echo -e "\n>>> Phase 1: 环境准备 (已跳过) <<<"
fi

# Phase 2: 能力评测
if [ "$SKIP_EVAL" = false ]; then
    echo -e "\n>>> Phase 2: 能力评测 <<<"
    $PYTHON -m src.cli eval run --config "$CONFIG" --models all --tasks all || {
        echo "ERROR: 能力评测失败"
        exit 1
    }
else
    echo -e "\n>>> Phase 2: 能力评测 (已跳过) <<<"
fi

# Phase 3: 性能评测
if [ "$SKIP_PERF" = false ]; then
    echo -e "\n>>> Phase 3: 性能评测 <<<"
    $PYTHON -m src.cli perf run --config "$CONFIG" --models all --hardware local || {
        echo "ERROR: 性能评测失败"
        exit 1
    }
else
    echo -e "\n>>> Phase 3: 性能评测 (已跳过) <<<"
fi

# Phase 4: 报告生成
echo -e "\n>>> Phase 4: 报告生成 <<<"
$PYTHON -m src.cli report generate --config "$CONFIG" --results-dir ./results --output ./report.md --format md || {
    echo "ERROR: 报告生成失败"
    exit 1
}
$PYTHON -m src.cli report generate --config "$CONFIG" --results-dir ./results --output ./report.json --format json || {
    echo "ERROR: 报告生成失败"
    exit 1
}

echo -e "\n========================================="
echo "  评测完成！"
echo "  报告: report.md / report.json / results/"
echo "========================================="
