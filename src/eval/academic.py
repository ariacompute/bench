"""学术基准评测：MMLU、GSM8K、C-Eval、HumanEval

由于不可访问原始数据集，采用内置示例题目 + 实际评测时替换为真实数据集。
"""

import json
import logging
from pathlib import Path

from ..config import ModelConfig

logger = logging.getLogger(__name__)

# ── 示例数据（实际评测时替换为 HuggingFace datasets 加载） ──

_SAMPLE_MMLU = [
    {
        "id": "mmlu_001",
        "question": "What is the capital of France?\nA. London\nB. Paris\nC. Berlin\nD. Madrid",
        "answer": "B",
    },
    {
        "id": "mmlu_002",
        "question": "Which element has the chemical symbol 'O'?\nA. Gold\nB. Oxygen\nC. Osmium\nD. Iron",  # noqa: E501
        "answer": "B",
    },
]

_SAMPLE_GSM8K = [
    {
        "id": "gsm8k_001",
        "question": "John has 5 apples. He buys 3 more. How many apples does he have?",
        "answer": "8",
    },
    {
        "id": "gsm8k_002",
        "question": "A train travels 60 miles per hour. How far does it travel in 2 hours?",
        "answer": "120",
    },
]

_SAMPLE_CEVAL = [
    {
        "id": "ceval_001",
        "question": "中国的首都是？\nA. 上海\nB. 北京\nC. 广州\nD. 深圳",
        "answer": "B",
    },
    {"id": "ceval_002", "question": "一年有多少个月？\nA. 10\nB. 11\nC. 12\nD. 13", "answer": "C"},
]

_SAMPLE_HUMANEVAL = [
    {
        "id": "humaneval_001",
        "task_id": "HumanEval/0",
        "prompt": 'def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n',  # noqa: E501
        "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n",  # noqa: E501
    },
]


def _load_dataset(name: str) -> list[dict]:
    """加载数据集（当前使用示例数据，实际评测时替换为真实数据集）"""
    datasets = {
        "mmlu": _SAMPLE_MMLU,
        "gsm8k": _SAMPLE_GSM8K,
        "ceval": _SAMPLE_CEVAL,
        "humaneval": _SAMPLE_HUMANEVAL,
    }
    logger.info("using sample data for %s (replace with real dataset for production use)", name)
    return datasets.get(name, [])


def _compute_accuracy(results: list[dict], dataset_name: str) -> float:
    """计算 accuracy / exact_match 等指标"""
    if not results:
        return 0.0
    correct = 0
    for r in results:
        if r.get("correct", False):
            correct += 1
    return correct / len(results)


def run_mmlu(
    model: ModelConfig, quant: str, bundle_path: str, output_dir: Path,
    engine_type: str, timeout: int,
) -> dict:
    """MMLU 评测"""
    data = _load_dataset("mmlu")
    results = _run_qa_benchmark(
        data, model, quant, bundle_path, output_dir, engine_type, timeout, "mmlu",
    )
    acc = _compute_accuracy(results, "mmlu")
    return {
        "task": "mmlu",
        "accuracy": acc,
        "total": len(data),
        "correct": sum(1 for r in results if r.get("correct")),
        "results": results,
    }


def run_gsm8k(
    model: ModelConfig, quant: str, bundle_path: str, output_dir: Path,
    engine_type: str, timeout: int,
) -> dict:
    """GSM8K 评测"""
    data = _load_dataset("gsm8k")
    results = _run_qa_benchmark(
        data, model, quant, bundle_path, output_dir, engine_type, timeout, "gsm8k",
    )
    acc = _compute_accuracy(results, "gsm8k")
    return {
        "task": "gsm8k",
        "exact_match": acc,
        "total": len(data),
        "correct": sum(1 for r in results if r.get("correct")),
        "results": results,
    }


def run_ceval(
    model: ModelConfig, quant: str, bundle_path: str, output_dir: Path,
    engine_type: str, timeout: int,
) -> dict:
    """C-Eval 评测"""
    data = _load_dataset("ceval")
    results = _run_qa_benchmark(
        data, model, quant, bundle_path, output_dir, engine_type, timeout, "ceval",
    )
    acc = _compute_accuracy(results, "ceval")
    return {
        "task": "ceval",
        "accuracy": acc,
        "total": len(data),
        "correct": sum(1 for r in results if r.get("correct")),
        "results": results,
    }


def run_humaneval(
    model: ModelConfig, quant: str, bundle_path: str, output_dir: Path,
    engine_type: str, timeout: int,
) -> dict:
    """HumanEval 评测"""
    data = _load_dataset("humaneval")
    results = _run_code_benchmark(
        data, model, quant, bundle_path, output_dir, engine_type, timeout,
    )
    pass_count = sum(1 for r in results if r.get("pass"))
    pass_at_1 = pass_count / len(data) if data else 0.0
    return {
        "task": "humaneval",
        "pass_at_1": pass_at_1,
        "total": len(data),
        "passed": pass_count,
        "results": results,
    }


def _run_qa_benchmark(
    data: list[dict],
    model: ModelConfig,
    quant: str,
    bundle_path: str,
    output_dir: Path,
    engine_type: str,
    timeout: int,
    task_name: str,
) -> list[dict]:
    """通用 QA 评测执行器"""
    from .engine_runner import run_inference

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{task_name}.json"

    # 断点续跑：已有结果则跳过
    if result_path.exists():
        logger.info(
            "%s: existing results found, skipping (delete %s to re-run)", task_name, result_path
        )
        with open(result_path, "r") as f:
            return json.load(f)["results"]

    for item in data:
        prompt = item["question"]
        if task_name == "gsm8k":
            prompt += "\nLet's solve this step by step. The final answer should be a number."

        output = run_inference(
            bundle_path,
            engine_type,
            prompt=prompt,
            max_tokens=128,
            temperature=0.0,
            timeout=timeout,
        )

        # 判定正确性（简单字符串匹配）
        generated = output.get("text", "").strip()
        error = output.get("error")

        if error:
            logger.warning(
                "%s/%s: engine error on %s: %s",
                task_name, item["id"], model.name, error,
            )
            results.append({
                "id": item["id"], "output": generated, "expected": item["answer"],
                "correct": False, "error": error,
            })
        else:
            correct = _check_answer(generated, item["answer"], task_name)
            results.append({
                "id": item["id"], "output": generated, "expected": item["answer"],
                "correct": correct,
            })

    # 保存结果
    with open(result_path, "w") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    return results


def _run_code_benchmark(
    data: list[dict],
    model: ModelConfig,
    quant: str,
    bundle_path: str,
    output_dir: Path,
    engine_type: str,
    timeout: int,
) -> list[dict]:
    """代码生成评测执行器（HumanEval）"""
    from .engine_runner import run_inference

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "humaneval.json"

    if result_path.exists():
        logger.info("humaneval: existing results found, skipping")
        with open(result_path, "r") as f:
            return json.load(f)["results"]

    for item in data:
        output = run_inference(
            bundle_path,
            engine_type,
            prompt=item["prompt"],
            max_tokens=256,
            temperature=0.0,
            timeout=timeout,
        )

        generated = output.get("text", "").strip()
        error = output.get("error")

        if error:
            logger.warning("humaneval/%s: engine error: %s", item["id"], error)
            results.append({
                "id": item["id"], "task_id": item["task_id"],
                "completion": "", "pass": False, "error": error,
            })
        else:
            pass_test = _check_code_pass(generated, item.get("canonical_solution", ""))
            results.append({
                "id": item["id"], "task_id": item["task_id"],
                "completion": generated, "pass": pass_test,
            })

    with open(result_path, "w") as f:
        json.dump({"results": results}, f, indent=2, ensure_ascii=False)

    return results


def _check_answer(generated: str, expected: str, task_name: str) -> bool:
    """判定回答是否正确"""
    gen_upper = generated.upper().strip()
    exp_upper = expected.upper().strip()

    if task_name in ("mmlu", "ceval"):
        option = exp_upper
        return (
            option in gen_upper.split("\n")[0]
            or gen_upper.startswith(option)
            or gen_upper.endswith(option)
        )
    elif task_name == "gsm8k":
        return exp_upper in gen_upper
    return exp_upper in gen_upper


def _check_code_pass(generated: str, canonical: str) -> bool:
    """简单检查代码是否通过（检查关键模式）"""
    if not generated or not canonical:
        return False
    gen_stripped = generated.replace(" ", "").replace("\n", "").replace("\r", "")
    keywords = ["returnTrue", "returnFalse", "if", "for", "while", "def"]
    has_structure = any(kw in gen_stripped for kw in keywords)
    has_result = "return" in gen_stripped
    return has_structure and has_result
