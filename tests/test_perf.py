"""perf 模块单元测试"""

from unittest.mock import patch

from src.config import ModelConfig
from src.perf.power import _has_rapl, _read_rapl_energy
from src.perf.size import measure_size
from src.perf.speed import _generate_prompt

# ── speed ──


def test_generate_prompt():
    prompt = _generate_prompt(512)
    assert len(prompt) > 0


@patch("src.eval.engine_runner.run_inference")
def test_measure_speed_basic(mock_run_inference, tmp_path):
    from src.perf.speed import measure_speed

    mock_run_inference.return_value = {
        "text": "hello world",
        "tokens": 10,
        "finish_reason": "stop",
        "ttft_ms": 50.0,
        "tokens_per_second": 20.0,
    }

    model = ModelConfig("test", "test", "hf/test", engine="aria", quantizations=["q4_channel"])
    result = measure_speed(model, "q4_channel", "/fake/bundle", tmp_path, "aria", 600)

    assert "ttft_ms" in result
    assert "tokens_per_second" in result
    assert "512" in result["ttft_ms"]


@patch("src.eval.engine_runner.run_inference")
def test_measure_speed_engine_error(mock_run_inference, tmp_path):
    from src.perf.speed import measure_speed

    # 模拟错误：返回带 error 字段的 dict（不再抛异常）
    mock_run_inference.return_value = {
        "text": "",
        "tokens": 0,
        "finish_reason": "error",
        "error": "test error",
    }

    model = ModelConfig("test", "test", "hf/test", engine="libama_cpp", quantizations=["q4_k_m"])
    result = measure_speed(model, "q4_k_m", "/fake/bundle", tmp_path, "llama_cpp", 600)

    # 全部分失败时指标为 0（但不抛异常）
    for inp in ["512", "1024", "2048"]:
        assert result["ttft_ms"][inp] == 0.0
        assert result["tokens_per_second"][inp] == 0.0


# ── memory ──


@patch("src.eval.engine_runner.run_inference")
def test_measure_memory(mock_run_inference, tmp_path):
    from src.perf.memory import measure_memory

    mock_run_inference.return_value = {"text": "ok", "tokens": 5, "finish_reason": "stop"}

    model = ModelConfig("test", "test", "hf/test", engine="aria", quantizations=["q4_channel"])
    result = measure_memory(model, "q4_channel", "/fake/bundle", tmp_path, "aria", 600)

    assert "peak_rss_mb" in result
    assert result["peak_rss_mb"] >= 0


# ── size ──


def test_measure_size(tmp_path):
    bundle_dir = tmp_path / "bundles" / "test" / "q4_channel"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "weight.bin").write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2 MB
    (bundle_dir / "config.json").write_text("{}")

    model = ModelConfig("test", "test", "hf/test", engine="aria", quantizations=["q4_channel"])
    result = measure_size(model, "q4_channel", str(bundle_dir), tmp_path)

    assert result["model_size_mb"] > 0
    assert "weight.bin" in result["files"]


def test_measure_size_missing_path(tmp_path):
    model = ModelConfig("test", "test", "hf/test", engine="aria", quantizations=["q4_channel"])
    result = measure_size(model, "q4_channel", "/nonexistent", tmp_path)

    assert result["model_size_mb"] == 0.0


# ── power ──


def test_has_rapl():
    result = _has_rapl()
    assert isinstance(result, bool)


def test_read_rapl_energy_returns_none_when_missing():
    with patch("src.perf.power.Path.exists", return_value=False):
        result = _read_rapl_energy(999)
        assert result is None


@patch("src.perf.power._has_rapl")
def test_measure_power_unavailable(mock_rapl, tmp_path):
    from src.perf.power import measure_power

    mock_rapl.return_value = False
    model = ModelConfig("test", "test", "hf/test", engine="aria", quantizations=["q4_channel"])
    result = measure_power(model, "q4_channel", "/fake/bundle", tmp_path)

    assert result["available"] is False
    assert result["power_watts"] is None
