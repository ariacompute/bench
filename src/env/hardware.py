"""硬件环境检测"""

import logging
from typing import Any

import psutil

logger = logging.getLogger(__name__)


def _detect_gpu() -> dict[str, Any]:
    """检测 GPU 信息"""
    result: dict[str, Any] = {"detected": False}
    try:
        import subprocess

        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            gpus = []
            for line in proc.stdout.strip().split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpus.append({"name": parts[0], "memory_mb": int(float(parts[1]))})
            if gpus:
                result = {"detected": True, "count": len(gpus), "devices": gpus}
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return result


def detect_hardware() -> dict[str, Any]:
    """检测当前硬件环境

    Returns:
        {
            "cpu": str,
            "cpu_cores": int,
            "ram_total_mb": int,
            "gpu": {...},
        }
    """
    cpu_info = ""
    try:
        import cpuinfo

        cpu_info = cpuinfo.get_cpu_info().get("brand_raw", "")
    except Exception:
        cpu_info = "Unknown"

    return {
        "cpu": cpu_info,
        "cpu_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "ram_total_mb": round(psutil.virtual_memory().total / (1024 * 1024)),
        "gpu": _detect_gpu(),
    }
