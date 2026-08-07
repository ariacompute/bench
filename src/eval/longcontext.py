"""长文本评测：Needle-in-Haystack"""

import json
import logging
import random
import string
from pathlib import Path

from ..config import ModelConfig

logger = logging.getLogger(__name__)


def _generate_filler_text(length: int) -> str:
    """生成无意义的填充文本"""
    words = [
        f"section_{i}_" + "".join(random.choices(string.ascii_lowercase, k=8))
        for i in range(length // 20)
    ]
    return " ".join(words)


def _insert_needle(text: str, needle: str, depth_percent: int) -> str:
    """在指定深度百分比位置插入 needle"""
    insert_pos = int(len(text) * depth_percent / 100)
    space_idx = text.find(" ", insert_pos)
    if space_idx == -1:
        space_idx = insert_pos
    return text[:space_idx] + f" {needle} " + text[space_idx:]


def run_needle_in_haystack(
    model: ModelConfig,
    quant: str,
    bundle_path: str,
    output_dir: Path,
    engine_type: str,
    timeout: int,
    context_lengths: list[int] | None = None,
    depth_percent: list[int] | None = None,
) -> dict:
    """Needle-in-Haystack 评测

    Args:
        model: 模型配置
        quant: 量化级别
        bundle_path: 模型本地路径
        output_dir: 输出目录
        engine_type: 推理引擎类型 (aria / llama_cpp / transformers)
        timeout: 推理超时秒数
        context_lengths: 上下文长度列表
        depth_percent: 插入深度百分比列表
    """
    from .engine_runner import run_inference

    if context_lengths is None:
        context_lengths = [4096, 8192, 16384, 32768]
    if depth_percent is None:
        depth_percent = [0, 10, 20, 50, 80, 90, 100]

    needle = f"The special passphrase is: ARIABENCH-{random.randint(10000, 99999)}"
    retrieval_prompt = (
        "What is the special passphrase mentioned in the text? Answer only with the passphrase."
    )

    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "needle_results.json"

    # 断点续跑
    completed: set = set()
    if result_path.exists():
        try:
            with open(result_path, "r") as f:
                existing = json.load(f)
            for r in existing.get("results", []):
                completed.add((r["context_length"], r["depth"]))
            logger.info("needle: resuming, %d combinations already completed", len(completed))
        except (json.JSONDecodeError, KeyError):
            pass

    total = len(context_lengths) * len(depth_percent)
    done = 0
    for ctx_len in context_lengths:
        if ctx_len > model.max_context:
            logger.info(
                "needle: skipping ctx_len=%d (exceeds model max %d)", ctx_len, model.max_context
            )
            continue
        for depth in depth_percent:
            if (ctx_len, depth) in completed:
                done += 1
                continue

            # 构造输入
            filler = _generate_filler_text(ctx_len)
            text_with_needle = _insert_needle(filler, needle, depth)
            prompt = text_with_needle + "\n\n" + retrieval_prompt

            # 估算 token 数并检查是否超限
            est_tokens = len(prompt.split()) * 1.3
            if est_tokens > model.max_context:
                logger.info(
                    "needle: skipping ctx_len=%d depth=%d (est %.0f tokens exceeds max %d)",
                    ctx_len, depth, est_tokens, model.max_context,
                )
                continue

            output = run_inference(
                bundle_path,
                engine_type,
                prompt=prompt,
                max_tokens=32,
                temperature=0.0,
                timeout=timeout,
            )

            generated = output.get("text", "").strip()
            error = output.get("error")
            retrieved = needle.lower() in generated.lower()

            results.append({
                "context_length": ctx_len,
                "depth": depth,
                "retrieved": retrieved,
                "output_prefix": generated[:100],
            })
            if error:
                results[-1]["error"] = error

            done += 1
            logger.info(
                "needle [%d/%d]: ctx=%d depth=%d → %s",
                done, total, ctx_len, depth,
                "✓" if retrieved else "✗",
            )

    accuracy = sum(1 for r in results if r.get("retrieved")) / len(results) if results else 0.0

    result_data = {
        "task": "needle_in_haystack",
        "accuracy": accuracy,
        "total_tests": len(results),
        "successful": sum(1 for r in results if r.get("retrieved")),
        "results": results,
    }

    with open(result_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    return result_data
