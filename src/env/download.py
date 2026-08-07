"""模型下载与缓存模块。

根据 engine 类型区分下载策略:
  - aria:   从 HuggingFace (ariacompute/*) 下载 Aria 量化模型包
  - llama_cpp: 从 HuggingFace 下载原始模型 + GGUF 转换提示
  - transformers: 使用 HF transformers 自动下载
"""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404
from typing import Optional

from .model_state import DownloadStatus

logger = logging.getLogger(__name__)

# 模型存储根目录，可通过环境变量 MODEL_STORE 覆盖
DEFAULT_MODEL_STORE = os.path.join(
    os.path.expanduser("~"), ".cache", "benchmark_models"
)


def _model_store() -> str:
    return os.environ.get("BENCHMARK_MODEL_STORE", DEFAULT_MODEL_STORE)


def _model_dir(source: str) -> str:
    """将 HF repo_id 或 ariacompute/ 前缀转为本地目录名。"""
    return os.path.join(_model_store(), source.replace("/", "_"))


# ── 下载器注册表 ──


def download_aria(source: str, force: bool = False) -> DownloadStatus:
    """下载 Aria 量化模型包 (ariacompute/ engine 专用)。

    Aria 模型以 quant bundle 形式发布在 HF ariacompute namespace 下。
    使用 huggingface_hub snapshot_download 拉取全量文件。
    """
    target = _model_dir(source)

    if os.path.isdir(target) and not force:
        logger.info("Aria 模型已缓存: %s → %s", source, target)
        return DownloadStatus.MODEL_READY

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub 未安装，降级到 git-lfs 克隆")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        repo_url = f"https://huggingface.co/{source}"
        subprocess.run(  # nosec B603
            ["git", "lfs", "clone", repo_url, target],
            check=True,
            timeout=600,
        )
        logger.info("Aria 模型下载完成 (git-lfs): %s", target)
        return DownloadStatus.MODEL_READY

    logger.info("下载 Aria 量化模型: %s", source)
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=source,
        local_dir=target,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info("Aria 模型下载完成: %s → %s", source, target)
    return DownloadStatus.MODEL_READY


def download_llama_cpp(source: str, quant: str, force: bool = False) -> DownloadStatus:
    """准备 llama.cpp 推理所需模型文件。

    策略:
    1. 优先直接下载预量化 GGUF（如 {source}-GGUF）。
    2. 若不存在 GGUF 仓库，下载原始 safetensors 并提示用户自行转换。

    Args:
        source: HF repo_id（如 Qwen/Qwen3-0.6B）
        quant: GGUF 量化级别（如 q4_k_m, q8_0）
    """
    target = _model_dir(source)

    # 如果已有 GGUF 文件，直接返回
    if os.path.isdir(target):
        gguf_files = [
            f for f in os.listdir(target)
            if f.endswith(".gguf") and quant.replace("_", "-") in f.lower()
        ]
        if gguf_files and not force:
            logger.info("llama.cpp GGUF 模型已缓存: %s (%s)", source, quant)
            return DownloadStatus.MODEL_READY

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        logger.warning("huggingface_hub 未安装，无法下载模型")
        return DownloadStatus.DOWNLOAD_FAILED

    # 尝试 GGUF 仓库
    gguf_repo = f"{source}-GGUF"
    try:
        files = list_repo_files(gguf_repo)
        gguf_file = next(
            (f for f in files if f.endswith(".gguf") and quant.replace("_", "-") in f.lower()),
            None,
        )
        if gguf_file:
            os.makedirs(target, exist_ok=True)
            hf_hub_download(
                repo_id=gguf_repo,
                filename=gguf_file,
                local_dir=target,
            )
            logger.info("GGUF 模型下载完成: %s/%s → %s", gguf_repo, gguf_file, target)
            return DownloadStatus.MODEL_READY
    except Exception as e:
        logger.debug("GGUF 仓库不存在或下载失败: %s, %s", gguf_repo, e)

    # 降级: 下载原始模型
    try:
        from huggingface_hub import snapshot_download

        os.makedirs(target, exist_ok=True)
        snapshot_download(
            repo_id=source,
            local_dir=target,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info(
            "原始模型下载完成: %s → %s\n"
            "⚠ 请使用 convert_hf_to_gguf.py 转换为 GGUF 格式: "
            "python convert_hf_to_gguf.py %s --outtype %s",
            source, target, target, quant,
        )
        return DownloadStatus.MODEL_READY
    except Exception as e:
        logger.error("下载失败: %s, %s", source, e)
        return DownloadStatus.DOWNLOAD_FAILED


def download_transformers(source: str, force: bool = False) -> DownloadStatus:
    """准备 transformers 推理所需模型文件。

    使用 HF transformers 自动下载机制（snapshot_download）。
    """
    target = _model_dir(source)

    if os.path.isdir(target) and not force:
        has_config = os.path.isfile(os.path.join(target, "config.json"))
        if has_config:
            logger.info("transformers 模型已缓存: %s → %s", source, target)
            return DownloadStatus.MODEL_READY

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.warning("huggingface_hub 未安装，无法下载模型")
        return DownloadStatus.DOWNLOAD_FAILED

    logger.info("下载 transformers 模型: %s", source)
    os.makedirs(target, exist_ok=True)
    snapshot_download(
        repo_id=source,
        local_dir=target,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    logger.info("transformers 模型下载完成: %s → %s", source, target)
    return DownloadStatus.MODEL_READY


# ── 下载调度 ──

def _dl_llama(source, quant=None, force=False):
    return download_llama_cpp(source, quant or "q4_k_m", force)


def _dl_transformers(source, quant=None, force=False):
    return download_transformers(source, force)


DOWNLOAD_MAP = {
    "aria": download_aria,
    "llama_cpp": _dl_llama,
    "transformers": _dl_transformers,
}


def download_model(
    engine: str,
    source: str,
    quantization: Optional[str] = None,
    force: bool = False,
) -> DownloadStatus:
    """根据引擎类型下载对应格式的模型文件。

    Args:
        engine: 引擎类型 (aria / llama_cpp / transformers)
        source: HF repo_id 或 ariacompute/ 前缀
        quantization: 量化级别（aria/llama_cpp 使用）
        force: 是否强制重新下载

    Returns:
        DownloadStatus 状态枚举。
    """
    downloader = DOWNLOAD_MAP.get(engine)
    if downloader is None:
        logger.error("不支持的引擎下载器: %s", engine)
        return DownloadStatus.DOWNLOAD_FAILED

    if engine == "aria":
        return downloader(source, force)  # type: ignore[call-arg]
    else:
        return downloader(source, quant=quantization, force=force)  # type: ignore[call-arg]


def get_model_local_path(source: str) -> str:
    """获取模型在本地缓存的路径。"""
    return _model_dir(source)
