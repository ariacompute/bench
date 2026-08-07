"""功耗测量（可选，仅 Intel RAPL / Linux PC）"""

import logging
import time
from pathlib import Path
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger(__name__)

RAPL_BASE = "/sys/class/powercap/intel-rapl"


def _has_rapl() -> bool:
    """检查 Intel RAPL 是否可用"""
    return Path(RAPL_BASE).exists()


def _read_rapl_energy(package_index: int = 0) -> float | None:
    """读取 Intel RAPL energy_uj（微焦耳）"""
    try:
        energy_path = Path(RAPL_BASE) / f"intel-rapl:{package_index}" / "energy_uj"
        if not energy_path.exists():
            return None
        with open(energy_path, "r") as f:
            return float(f.read().strip()) / 1_000_000  # 转换为焦耳
    except (OSError, ValueError) as e:
        logger.warning("failed to read RAPL energy[%d]: %s", package_index, e)
        return None


def measure_power(
    model: ModelConfig, quant: str, bundle_path: str, output_dir: Path
) -> dict[str, Any]:
    """测量推理功耗（仅 Intel RAPL 路径）

    Returns:
        {
            "available": bool,
            "energy_joules": float | None,
            "power_watts": float | None,
        }
    """
    if not _has_rapl():
        logger.info("power measurement: Intel RAPL not available, skipping")
        return {"available": False, "energy_joules": None, "power_watts": None}

    from ..eval.engine_runner import run_inference

    energy_before = _read_rapl_energy()
    t_start = time.perf_counter()

    try:
        prompt = (
            "Power measurement benchmark. " + "The quick brown fox jumps over the lazy dog. " * 50
        )
        run_inference(
            bundle_path,
            model.engine,
            prompt=prompt,
            max_tokens=64,
            temperature=0.0,
            timeout=300,
        )
    except Exception as e:
        logger.warning("power measurement: engine error: %s", e)

    t_end = time.perf_counter()
    energy_after = _read_rapl_energy()

    if energy_before is not None and energy_after is not None:
        energy_j = energy_after - energy_before
        elapsed_s = t_end - t_start
        power_w = round(energy_j / elapsed_s, 2) if elapsed_s > 0 else None
        return {"available": True, "energy_joules": round(energy_j, 4), "power_watts": power_w}

    return {"available": True, "energy_joules": None, "power_watts": None}
