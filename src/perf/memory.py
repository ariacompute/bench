"""内存占用测量：运行时峰值 RAM"""

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from ..config import ModelConfig

logger = logging.getLogger(__name__)


class _MemorySampler:
    """后台线程周期采样内存使用"""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self._peak_rss: int = 0
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._peak_rss = 0
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        return {"peak_rss_mb": round(self._peak_rss / (1024 * 1024), 2)}

    def _sample_loop(self) -> None:
        while self._running:
            try:
                process = psutil.Process(os.getpid())
                rss = process.memory_info().rss
                for child in process.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                if rss > self._peak_rss:
                    self._peak_rss = rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(self.interval)


def measure_memory(
    model: ModelConfig,
    quant: str,
    bundle_path: str,
    output_dir: Path,
    engine_type: str,
    timeout: int,
) -> dict[str, Any]:
    """测量运行时峰值内存

    Returns:
        {
            "peak_rss_mb": float,
        }
    """
    from ..eval.engine_runner import run_inference

    sampler = _MemorySampler(interval=0.05)
    sampler.start()

    try:
        prompt = (
            "Benchmark memory measurement. " + "The quick brown fox jumps over the lazy dog. " * 50
        )
        run_inference(
            bundle_path,
            engine_type,
            prompt=prompt,
            max_tokens=64,
            temperature=0.0,
            timeout=timeout,
        )
    except Exception:
        logger.warning("memory measurement: engine error")
    finally:
        result = sampler.stop()

    return {"peak_rss_mb": result["peak_rss_mb"]}
