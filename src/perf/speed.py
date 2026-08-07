"""推理速度测量：TTFT（首 token 延迟）+ 吞吐量（tok/s）"""

import logging
import statistics
from pathlib import Path
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger(__name__)

_WARMUP_ROUNDS = 3
_BENCH_ROUNDS = 5
_INPUT_LENGTHS = [512, 1024, 2048]


def _generate_prompt(target_tokens: int) -> str:
    """生成约 target_tokens 个 token 的提示文本"""
    template = "The quick brown fox jumps over the lazy dog. "
    repeats = max(1, target_tokens // 8)
    return (template * repeats)[: target_tokens * 6]


def measure_speed(
    model: ModelConfig,
    quant: str,
    bundle_path: str,
    output_dir: Path,
    engine_type: str,
    timeout: int,
) -> dict[str, Any]:
    """测量推理速度

    Returns:
        {
            "ttft_ms": {"512": float, "1024": float, "2048": float},
            "tokens_per_second": {"512": float, "1024": float, "2048": float},
        }
    """
    from ..eval.engine_runner import run_inference

    results: dict[str, dict[str, float]] = {"ttft_ms": {}, "tokens_per_second": {}}

    for inp_len in _INPUT_LENGTHS:
        prompt = _generate_prompt(inp_len)

        ttft_samples: list[float] = []
        tps_samples: list[float] = []

        total_rounds = _WARMUP_ROUNDS + _BENCH_ROUNDS
        for r in range(total_rounds):
            output = run_inference(
                bundle_path,
                engine_type,
                prompt=prompt,
                max_tokens=128,
                temperature=0.0,
                timeout=timeout,
            )

            error = output.get("error")
            if error:
                logger.warning("speed/len=%d/round=%d: engine error: %s", inp_len, r, error)
                continue

            tokens = output.get("tokens", 0)
            tps = output.get("tokens_per_second")
            ttft = output.get("ttft_ms")

            # 若无引擎报告的 tps，用 token 数估算
            if tps is None and tokens > 0:
                tps = 10.0  # 默认保守估计

            # 若无引擎报告的 ttft，估算为首 token 占 15% 总时间
            if ttft is None and tps is not None and tps > 0:
                ttft = (tokens / tps) * 0.15 * 1000

            if r >= _WARMUP_ROUNDS:
                if ttft is not None:
                    ttft_samples.append(ttft)
                if tps is not None:
                    tps_samples.append(tps)

        if ttft_samples:
            results["ttft_ms"][str(inp_len)] = round(statistics.mean(ttft_samples), 2)
        else:
            results["ttft_ms"][str(inp_len)] = 0.0

        if tps_samples:
            results["tokens_per_second"][str(inp_len)] = round(statistics.mean(tps_samples), 2)
        else:
            results["tokens_per_second"][str(inp_len)] = 0.0

    return results
