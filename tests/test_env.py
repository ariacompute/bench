"""env 模块单元测试"""

from unittest.mock import MagicMock, patch

from src.config import EngineConfig
from src.env.download import download_model, get_model_local_path
from src.env.engine import (
    EngineCheckResult,
    EngineInstallStatus,
    check_all_engines,
    check_aria_engine,
    check_engine,
    check_llama_cpp,
)
from src.env.hardware import detect_hardware
from src.env.model_state import DownloadStatus, ModelState, load_state, save_state

# ── model_state ──


def test_model_state_roundtrip(tmp_path):
    state = ModelState()
    state.set_downloaded("test-model", {"q4_channel": {"path": "/tmp/bundle"}})
    save_state(tmp_path, state)

    loaded = load_state(tmp_path)
    assert loaded.get("test-model")["downloaded"] is True
    assert "q4_channel" in loaded.get("test-model")["bundles"]


def test_model_state_failed(tmp_path):
    state = ModelState()
    state.set_failed("bad-model", "download error")
    assert state.get("bad-model")["downloaded"] is False
    assert state.get("bad-model")["error"] == "download error"


def test_load_corrupted_state(tmp_path):
    spath = tmp_path / "state.json"
    spath.write_text("not valid json")
    state = load_state(tmp_path)
    assert not state.models


def test_load_missing_state(tmp_path):
    state = load_state(tmp_path)
    assert not state.models


# ── engine 检测 ──


@patch("src.env.engine.shutil.which", return_value=None)
def test_aria_engine_not_found(mock_which):
    result = check_aria_engine()
    assert result.engine_type == "aria"
    assert result.status == EngineInstallStatus.NOT_FOUND


@patch("src.env.engine.shutil.which", return_value="/usr/bin/aria-engine")
@patch("src.env.engine.subprocess.run")
def test_aria_engine_ok(mock_run, mock_which):
    mock_run.return_value = MagicMock(returncode=0, stdout=b"aria-engine v0.1.0\n", stderr=b"")
    result = check_aria_engine()
    assert result.status == EngineInstallStatus.OK
    assert "v0.1.0" in result.version_string


@patch("src.env.engine.shutil.which", return_value=None)
def test_llama_cpp_not_found(mock_which):
    result = check_llama_cpp()
    assert result.engine_type == "llama_cpp"
    assert result.status == EngineInstallStatus.NOT_FOUND


@patch("src.env.engine.shutil.which", return_value="/usr/local/bin/llama-cli")
@patch("src.env.engine.subprocess.run")
def test_llama_cpp_ok(mock_run, mock_which):
    mock_run.return_value = MagicMock(returncode=0, stdout=b"llama.cpp version b4000\n", stderr=b"")
    result = check_llama_cpp()
    assert result.status == EngineInstallStatus.OK


@patch("src.env.engine.check_transformers")
def test_transformers_installed(mock_check):
    mock_check.return_value = EngineCheckResult(
        engine_type="transformers",
        status=EngineInstallStatus.OK,
        version_string="4.40.0",
    )
    result = mock_check()
    assert result.status == EngineInstallStatus.OK
    assert result.version_string == "4.40.0"


def test_check_engine_dispatch():
    """check_engine 调度正确"""
    fake_aria = MagicMock()
    fake_aria.return_value = EngineCheckResult(
        engine_type="aria", status=EngineInstallStatus.OK, binary_path="/fake/aria-engine"
    )
    fake_llama = MagicMock()
    fake_llama.return_value = EngineCheckResult(
        engine_type="llama_cpp", status=EngineInstallStatus.NOT_FOUND
    )
    with patch.dict("src.env.engine.ENGINE_CHECK_MAP", {"aria": fake_aria, "llama_cpp": fake_llama}):
        result = check_engine("aria")
        assert result.status == EngineInstallStatus.OK
        result2 = check_engine("llama_cpp")
        assert result2.status == EngineInstallStatus.NOT_FOUND
        result3 = check_engine("nonexistent_engine")
        assert result3.status == EngineInstallStatus.ERROR


def test_check_all_engines():
    results = check_all_engines(["aria", "llama_cpp", "transformers"])
    assert len(results) == 3
    assert results[0].engine_type == "aria"
    assert results[1].engine_type == "llama_cpp"
    assert results[2].engine_type == "transformers"


# ── download 调度 ──


@patch("src.env.download.download_aria")
def test_download_model_aria_dispatches(mock_download):
    mock_download.return_value = DownloadStatus.MODEL_READY
    with patch.dict("src.env.download.DOWNLOAD_MAP", {"aria": mock_download}):
        result = download_model("aria", "ariacompute/test", "q4_channel")
        assert result == DownloadStatus.MODEL_READY
        mock_download.assert_called_once()


@patch("src.env.download.download_llama_cpp")
def test_download_model_llama_cpp_dispatches(mock_download):
    mock_download.return_value = DownloadStatus.MODEL_READY
    with patch.dict("src.env.download.DOWNLOAD_MAP", {"llama_cpp": mock_download}):
        result = download_model("llama_cpp", "Qwen/Qwen3-0.6B", "q4_k_m")
        assert result == DownloadStatus.MODEL_READY
        mock_download.assert_called_once()


@patch("src.env.download.download_transformers")
def test_download_model_transformers_dispatches(mock_download):
    mock_download.return_value = DownloadStatus.MODEL_READY
    with patch.dict("src.env.download.DOWNLOAD_MAP", {"transformers": mock_download}):
        result = download_model("transformers", "LiquidAI/LFM2-350M", quantization=None)
        assert result == DownloadStatus.MODEL_READY
        mock_download.assert_called_once()


def test_download_model_unknown_engine():
    result = download_model("unknown", "hf/test")
    assert result == DownloadStatus.DOWNLOAD_FAILED


def test_get_model_local_path():
    path = get_model_local_path("ariacompute/Qwen3-0.6B")
    assert "ariacompute_Qwen3-0.6B" in path


# ── EngineConfig ──


def test_engine_config_defaults():
    cfg = EngineConfig(engine_type="aria", source="https://test")
    assert cfg.engine_type == "aria"
    assert cfg.binary is None
    assert cfg.timeout_seconds == 3600


# ── hardware ──


def test_detect_hardware():
    hw = detect_hardware()
    assert "cpu" in hw
    assert "cpu_cores" in hw
    assert "ram_total_mb" in hw
    assert hw["ram_total_mb"] > 0
