"""模型体积测量：bundle 文件大小"""

import logging
from pathlib import Path
from typing import Any

from ..config import ModelConfig

logger = logging.getLogger(__name__)


def measure_size(
    model: ModelConfig, quant: str, bundle_path: str, output_dir: Path
) -> dict[str, Any]:
    """测量模型 bundle 体积

    Returns:
        {
            "model_size_mb": float,
            "files": {filename: size_mb},
        }
    """
    bp = Path(bundle_path)
    if not bp.exists():
        logger.warning("bundle path not found: %s", bp)
        return {"model_size_mb": 0.0, "files": {}}

    total_size = 0
    files: dict[str, float] = {}
    for f in bp.rglob("*"):
        if f.is_file():
            size = f.stat().st_size
            total_size += size
            files[f.name] = round(size / (1024 * 1024), 2)

    return {
        "model_size_mb": round(total_size / (1024 * 1024), 2),
        "files": files,
    }
