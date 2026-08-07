"""指令遵循评测：IFEval"""

import json
import logging
from pathlib import Path

from ..config import ModelConfig

logger = logging.getLogger(__name__)

# 示例 IFEval 数据
_SAMPLE_IFEVAL = [
    {
        "id": "ifeval_001",
        "prompt": "Write a response that contains exactly 3 sentences and mentions the word 'machine'.",  # noqa: E501
        "constraints": ["sentence_count=3", "contains=machine"],
    },
    {
        "id": "ifeval_002",
        "prompt": "Write a paragraph with at least 5 sentences about artificial intelligence, starting with the word 'Artificial'.",  # noqa: E501
        "constraints": ["sentence_count>=5", "startswith=Artificial"],
    },
]


def _load_ifeval() -> list[dict]:
    logger.info("using sample IFEval data (replace with real dataset for production use)")
    return _SAMPLE_IFEVAL


def _check_constraint(output: str, constraint: str) -> bool:
    """检查单个约束是否满足"""
    if constraint.startswith("sentence_count="):
        target = int(constraint.split("=")[1])
        sentences = [s for s in output.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        return len(sentences) == target
    elif constraint.startswith("sentence_count>="):
        target = int(constraint.split(">=")[1])
        sentences = [s for s in output.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        return len(sentences) >= target
    elif constraint.startswith("contains="):
        keyword = constraint.split("=")[1]
        return keyword.lower() in output.lower()
    elif constraint.startswith("startswith="):
        keyword = constraint.split("=")[1]
        return output.strip().lower().startswith(keyword.lower())
    else:
        return True  # 未知约束，默认通过


def run_ifeval(
    model: ModelConfig,
    quant: str,
    bundle_path: str,
    output_dir: Path,
    engine_type: str,
    timeout: int,
) -> dict:
    """IFEval 评测"""
    from .engine_runner import run_inference

    data = _load_ifeval()
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "ifeval.json"

    if result_path.exists():
        logger.info("ifeval: existing results found, skipping")
        with open(result_path, "r") as f:
            return json.load(f)

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
            logger.warning("ifeval/%s: engine error: %s", item["id"], error)
            results.append({
                "id": item["id"], "output": generated,
                "constraints": item["constraints"],
                "passed_all": False, "error": error,
            })
        else:
            passed = all(_check_constraint(generated, c) for c in item["constraints"])
            results.append({
                "id": item["id"], "output": generated,
                "constraints": item["constraints"],
                "passed_all": passed,
            })

    strict_acc = sum(1 for r in results if r["passed_all"]) / len(results) if results else 0.0

    result_data = {
        "task": "ifeval",
        "strict_accuracy": strict_acc,
        "total": len(data),
        "passed": sum(1 for r in results if r["passed_all"]),
        "results": results,
    }

    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    return result_data
