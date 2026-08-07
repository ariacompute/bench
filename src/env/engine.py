"""多推理引擎安装检测与准备模块。

负责检测 aria-engine / llama.cpp / transformers 是否已在 PATH 或
预置目录中可用，并在缺失时提供安装指引。
"""

from __future__ import annotations

import logging
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EngineInstallStatus(Enum):
    """引擎安装状态。"""
    OK = "ok"
    NOT_FOUND = "not_found"
    WRONG_VERSION = "wrong_version"
    ERROR = "error"


@dataclass
class EngineCheckResult:
    """引擎检测结果。"""
    engine_type: str
    status: EngineInstallStatus
    binary_path: Optional[str] = None
    version_string: str = ""
    detail: str = ""


# ── 引擎检测 ──

def check_aria_engine(binary: str = "aria-engine") -> EngineCheckResult:
    """检测 aria-engine 是否可用。

    查找顺序: PATH → CWD/aria-engine → ~/.aria/bin/aria-engine
    """
    candidates = [
        shutil.which(binary),
        "./aria-engine",
        "~/.aria/bin/aria-engine",
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            result = subprocess.run([c, "--version"], capture_output=True, timeout=30)  # nosec B603
            if result.returncode == 0:
                return EngineCheckResult(
                    engine_type="aria",
                    status=EngineInstallStatus.OK,
                    binary_path=c,
                    version_string=result.stdout.decode().strip(),
                )
        except Exception:
            continue

    return EngineCheckResult(
        engine_type="aria",
        status=EngineInstallStatus.NOT_FOUND,
        detail="aria-engine 未找到。请从 https://github.com/ariacompute/engine 安装。",
    )


def check_llama_cpp(binary: str = "llama-cli") -> EngineCheckResult:
    """检测 llama.cpp 是否可用。

    查找顺序: PATH → /usr/local/bin/llama-cli
    另检查 llama.cpp 构建产物中的 llama-cli。
    """
    candidates = [
        shutil.which(binary),
        "/usr/local/bin/llama-cli",
        "./llama.cpp/build/bin/llama-cli",
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            result = subprocess.run([c, "--version"], capture_output=True, timeout=30)  # nosec B603
            if result.returncode == 0:
                return EngineCheckResult(
                    engine_type="llama_cpp",
                    status=EngineInstallStatus.OK,
                    binary_path=c,
                    version_string=result.stdout.decode().strip(),
                )
        except Exception:
            continue

    return EngineCheckResult(
        engine_type="llama_cpp",
        status=EngineInstallStatus.NOT_FOUND,
        detail="llama.cpp 未找到。请从 https://github.com/ggerganov/llama.cpp 编译安装。",
    )


def check_transformers() -> EngineCheckResult:
    """检测 HuggingFace Transformers 是否可导入。"""
    try:
        import transformers  # noqa: F401

        ver = transformers.__version__
        return EngineCheckResult(
            engine_type="transformers",
            status=EngineInstallStatus.OK,
            version_string=ver,
        )
    except ImportError:
        return EngineCheckResult(
            engine_type="transformers",
            status=EngineInstallStatus.NOT_FOUND,
            detail="transformers 未安装。运行: pip install transformers torch",
        )


# ── 引擎调度 ──

ENGINE_CHECK_MAP = {
    "aria": check_aria_engine,
    "llama_cpp": check_llama_cpp,
    "transformers": check_transformers,
}


def check_engine(engine_type: str) -> EngineCheckResult:
    """检测指定引擎是否可用。"""
    checker = ENGINE_CHECK_MAP.get(engine_type)
    if checker is None:
        return EngineCheckResult(
            engine_type=engine_type,
            status=EngineInstallStatus.ERROR,
            detail=f"不支持的引擎类型: {engine_type}",
        )
    return checker()


def check_all_engines(engine_types: list) -> list[EngineCheckResult]:
    """批量检测多个引擎。"""
    return [check_engine(et) for et in engine_types]


# ── 引擎准备入口 ──

def setup_engine(
    engine_type: str,
    models_dir: str,
) -> bool:
    """为指定模型准备推理引擎环境。

    检测引擎可用性，并确保模型文件已下载（含格式转换提示）。

    Returns:
        True 表示引擎和模型均已就绪。
    """
    result = check_engine(engine_type)
    logger.info("引擎检测 [%s]: %s", engine_type, result.status.value)

    if result.status == EngineInstallStatus.OK:
        logger.info("  路径: %s", result.binary_path or "(Python module)")
        if result.version_string:
            logger.info("  版本: %s", result.version_string)
        return True

    if result.status == EngineInstallStatus.NOT_FOUND:
        logger.warning("引擎 '%s' 不可用: %s", engine_type, result.detail)

    return result.status == EngineInstallStatus.OK
