"""模型状态管理：持久化模型下载/量化状态，支持断点续跑"""

import json
import time
from enum import Enum
from pathlib import Path
from typing import Any


class DownloadStatus(Enum):
    """模型下载结果状态。"""
    MODEL_READY = "model_ready"
    DOWNLOAD_FAILED = "download_failed"


class ModelState:
    """单个模型的状态快照"""

    def __init__(self) -> None:
        self.models: dict[str, dict[str, Any]] = {}

    def get(self, model_name: str) -> dict[str, Any]:
        return self.models.get(model_name, {})

    def set_downloaded(self, model_name: str, bundles: dict[str, dict[str, Any]]) -> None:
        self.models[model_name] = {
            "downloaded": True,
            "bundles": bundles,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    def set_failed(self, model_name: str, error: str) -> None:
        self.models[model_name] = {
            "downloaded": False,
            "error": error,
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


def state_file_path(output_dir: Path) -> Path:
    return output_dir / "state.json"


def load_state(output_dir: Path) -> ModelState:
    """加载模型状态文件"""
    spath = state_file_path(output_dir)
    state = ModelState()
    if spath.exists():
        try:
            with open(spath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            state.models = raw.get("models", {})
        except (json.JSONDecodeError, KeyError):
            # 损坏的状态文件，忽略
            pass
    return state


def save_state(output_dir: Path, state: ModelState) -> None:
    """保存模型状态文件"""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(state_file_path(output_dir), "w", encoding="utf-8") as f:
        json.dump({"models": state.models}, f, indent=2, ensure_ascii=False)
