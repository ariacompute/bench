"""report 模块单元测试"""

import json

from src.config import ScoringConfig
from src.report.generator import (
    _calculate_score,
    _collect_results,
    _generate_markdown,
    generate_report,
)


def test_collect_results_empty(tmp_path):
    data = _collect_results(tmp_path)
    assert data["eval"] == {}
    assert data["perf"] == {}


def test_collect_results_with_data(tmp_path):
    eval_dir = tmp_path / "eval_summary.json"
    eval_dir.write_text(json.dumps({"model1": {"q4": {"mmlu": {"accuracy": 0.8}}}}))
    perf_dir = tmp_path / "perf_summary.json"
    perf_dir.write_text(
        json.dumps(
            {
                "model1": {
                    "q4": {
                        "speed": {"ttft_ms": {"1024": 100.0}},
                        "memory": {"peak_rss_mb": 500},
                        "size": {"model_size_mb": 200},
                    }
                }
            }
        )
    )

    data = _collect_results(tmp_path)
    assert "model1" in data["eval"]
    assert "model1" in data["perf"]


def test_calculate_score():
    scoring = ScoringConfig(
        capability_weight=0.40,
        efficiency_weight=0.25,
        longcontext_weight=0.10,
        instruction_weight=0.05,
    )

    eval_data = {
        "q4_channel": {
            "mmlu": {"accuracy": 0.8},
            "gsm8k": {"exact_match": 0.7},
            "needle_in_haystack": {"accuracy": 0.9},
            "ifeval": {"strict_accuracy": 0.6},
        }
    }

    perf_data = {
        "q4_channel": {
            "speed": {"tokens_per_second": {"1024": 50.0}},
            "memory": {"peak_rss_mb": 500},
            "size": {"model_size_mb": 200},
        }
    }

    score = _calculate_score("test", eval_data, perf_data, scoring)
    assert "capability" in score
    assert "efficiency" in score
    assert "total" in score


def test_generate_markdown_with_engine_info():
    """Markdown 报告应包含引擎信息"""
    scoring = ScoringConfig()
    data = {
        "eval": {
            "test-model-aria": {
                "q4_channel": {
                    "mmlu": {"accuracy": 0.85},
                }
            },
            "test-model-native": {
                "q4_k_m": {
                    "mmlu": {"accuracy": 0.82},
                }
            },
        },
        "perf": {
            "test-model-aria": {
                "q4_channel": {
                    "speed": {"ttft_ms": {"1024": 50.0}, "tokens_per_second": {"1024": 40.0}},
                    "memory": {"peak_rss_mb": 300},
                    "size": {"model_size_mb": 150},
                }
            },
            "test-model-native": {
                "q4_k_m": {
                    "speed": {"ttft_ms": {"1024": 60.0}, "tokens_per_second": {"1024": 35.0}},
                    "memory": {"peak_rss_mb": 350},
                    "size": {"model_size_mb": 180},
                }
            },
        },
    }

    md = _generate_markdown(data, [], scoring)
    assert "端侧大模型综合评测报告" in md
    assert "test-model-aria" in md
    assert "test-model-native" in md
    assert "综合评分排行榜" in md
    # 引擎列显示 aria / llama_cpp
    assert "引擎" in md or any(engine in md for engine in ["aria", "llama_cpp"])


def test_generate_markdown_empty():
    scoring = ScoringConfig()
    md = _generate_markdown({"eval": {}, "perf": {}}, [], scoring)
    assert "综合评测报告" in md


def test_generate_report_json(tmp_path):
    eval_dir = tmp_path / "eval_summary.json"
    eval_dir.write_text("{}")
    perf_dir = tmp_path / "perf_summary.json"
    perf_dir.write_text("{}")

    output = tmp_path / "report.json"
    generate_report(None, tmp_path, output, fmt="json")

    assert output.exists()
    with open(output) as f:
        data = json.load(f)
    assert "eval" in data


def test_generate_report_markdown(tmp_path):
    eval_dir = tmp_path / "eval_summary.json"
    eval_dir.write_text("{}")
    perf_dir = tmp_path / "perf_summary.json"
    perf_dir.write_text("{}")

    output = tmp_path / "report.md"
    generate_report(None, tmp_path, output, fmt="md")

    assert output.exists()
    content = output.read_text()
    assert "综合评测报告" in content
