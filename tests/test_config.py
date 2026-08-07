"""配置模块单元测试"""

from pathlib import Path

import pytest
import yaml

from src.config import (
    ConfigError,
    EngineConfig,
    ModelConfig,
    load_config,
)


def _write_config(tmp: Path, data: dict) -> Path:
    config_path = tmp / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(data, f)
    return config_path


def _base_engine_section() -> dict:
    return {
        "aria": {"source": "https://github.com/ariacompute/engine", "binary": "aria-engine", "timeout_seconds": 600},
        "llama_cpp": {"source": "https://github.com/ggerganov/llama.cpp", "binary": "llama-cli", "timeout_seconds": 600},
        "transformers": {"source": "https://github.com/huggingface/transformers", "timeout_seconds": 600},
    }


# ── 多引擎配置加载 ──


def test_load_valid_config_with_engines(tmp_path):
    """加载含 engines 段的多引擎配置"""
    data = {
        "benchmark": {"name": "test", "version": "1.0", "output_dir": "./results"},
        "models": [
            {
                "name": "test-model-aria",
                "family": "test",
                "source": "ariacompute/test",
                "engine": "aria",
                "quantizations": ["q4_channel", "q8_channel"],
            }
        ],
        "engines": _base_engine_section(),
        "tasks": {"academic": ["mmlu"], "instruction": ["ifeval"]},
        "hardware": {"targets": [{"name": "local", "type": "x86-cpu"}]},
        "metrics": {"capability": ["mmlu_acc"], "performance": ["ttft_ms"]},
        "scoring": {"capability_weight": 0.4},
    }
    path = _write_config(tmp_path, data)
    config = load_config(str(path))
    assert config.name == "test"
    assert len(config.models) == 1
    assert config.models[0].name == "test-model-aria"
    assert config.models[0].engine == "aria"
    assert "aria" in config.engines
    assert config.engines["aria"].binary == "aria-engine"


def test_load_config_with_native_engine_models(tmp_path):
    """加载含原生引擎模型的配置"""
    data = {
        "benchmark": {"name": "test"},
        "models": [
            {
                "name": "qwen-aria",
                "family": "qwen",
                "source": "ariacompute/Qwen3-0.6B",
                "engine": "aria",
                "quantizations": ["q4_channel"],
            },
            {
                "name": "qwen-native",
                "family": "qwen",
                "source": "Qwen/Qwen3-0.6B",
                "engine": "llama_cpp",
                "quantizations": ["q4_k_m"],
                "compare_with": "qwen-aria",
            },
        ],
        "engines": _base_engine_section(),
    }
    path = _write_config(tmp_path, data)
    config = load_config(str(path))
    assert len(config.models) == 2
    assert config.models[0].engine == "aria"
    assert config.models[1].engine == "llama_cpp"
    assert config.models[1].compare_with == "qwen-aria"
    assert config.models[1].is_native is True


# ── engine 字段校验 ──


def test_engine_field_defaults_to_aria(tmp_path):
    """engine 字段缺失时默认 'aria'"""
    data = {
        "benchmark": {"name": "test"},
        "models": [{"name": "m", "family": "f", "source": "hf/f", "quantizations": ["q4_channel"]}],
        "engines": _base_engine_section(),
    }
    path = _write_config(tmp_path, data)
    config = load_config(str(path))
    assert config.models[0].engine == "aria"


def test_invalid_engine_type_raises(tmp_path):
    """非法的 engine 类型应抛出 ConfigError"""
    data = {
        "benchmark": {"name": "test"},
        "models": [{"name": "m", "family": "f", "source": "hf/f", "engine": "vllm", "quantizations": ["fp16"]}],
        "engines": _base_engine_section(),
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="未知引擎类型"):
        load_config(str(path))


def test_model_engine_not_in_engines_section(tmp_path):
    """模型引用的 engine 未在 engines 段定义时报错"""
    data = {
        "benchmark": {"name": "test"},
        "models": [{"name": "m", "family": "f", "source": "hf/f", "engine": "llama_cpp", "quantizations": ["q4_k_m"]}],
        "engines": {"aria": {"source": "https://github.com/ariacompute/engine"}},
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="引用了未在 engines 段定义的引擎"):
        load_config(str(path))


# ── compare_with 校验 ──


def test_compare_with_valid(tmp_path):
    """compare_with 指引合法模型"""
    data = {
        "benchmark": {"name": "test"},
        "models": [
            {"name": "a", "family": "f", "source": "hf/a", "engine": "aria", "quantizations": ["q4_channel"]},
            {"name": "b", "family": "f", "source": "hf/b", "engine": "llama_cpp", "quantizations": ["q4_k_m"], "compare_with": "a"},
        ],
        "engines": _base_engine_section(),
    }
    path = _write_config(tmp_path, data)
    config = load_config(str(path))
    assert config.models[1].compare_with == "a"


def test_compare_with_invalid_raises(tmp_path):
    """compare_with 指引不存在的模型时报错"""
    data = {
        "benchmark": {"name": "test"},
        "models": [
            {"name": "a", "family": "f", "source": "hf/a", "engine": "aria", "quantizations": ["q4_channel"], "compare_with": "xxxxx"},
        ],
        "engines": _base_engine_section(),
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="compare_with"):
        load_config(str(path))


# ── 错误场景 ──


def test_config_empty_file(tmp_path):
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("")
    with pytest.raises(ConfigError, match="为空"):
        load_config(str(config_path))


def test_config_missing_file():
    with pytest.raises(ConfigError, match="不存在"):
        load_config("/nonexistent/config.yaml")


def test_duplicate_model_names(tmp_path):
    """重复的模型名应报错"""
    data = {
        "benchmark": {"name": "test"},
        "models": [
            {"name": "dup", "family": "f", "source": "hf/a", "engine": "aria", "quantizations": ["q4_channel"]},
            {"name": "dup", "family": "f", "source": "hf/b", "engine": "llama_cpp", "quantizations": ["q4_k_m"]},
        ],
        "engines": _base_engine_section(),
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="重复的模型名"):
        load_config(str(path))


def test_model_missing_name(tmp_path):
    """缺少 name 字段的模型应报错"""
    data = {
        "benchmark": {"name": "test"},
        "models": [{"family": "f", "source": "hf/f", "quantizations": ["q4_channel"]}],
        "engines": _base_engine_section(),
    }
    path = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="缺少 name"):
        load_config(str(path))


# ── EngineConfig 数据类 ──


def test_engine_config_defaults():
    cfg = EngineConfig(engine_type="aria", source="https://test")
    assert cfg.engine_type == "aria"
    assert cfg.release == "latest"
    assert cfg.binary is None
    assert cfg.timeout_seconds == 3600


def test_engine_config_with_binary():
    cfg = EngineConfig(
        engine_type="llama_cpp",
        source="https://github.com/ggerganov/llama.cpp",
        binary="llama-cli",
        timeout_seconds=600,
    )
    assert cfg.binary == "llama-cli"
    assert cfg.timeout_seconds == 600


# ── ModelConfig 属性 ──


def test_model_is_native():
    m = ModelConfig("test", "f", "hf/f", engine="llama_cpp", quantizations=["q4_k_m"])
    assert m.is_native is True

    m2 = ModelConfig("test2", "f", "hf/f2", engine="aria", quantizations=["q4_channel"])
    assert m2.is_native is False


def test_model_is_vla():
    for family in ("openvla", "openpi", "lingbot"):
        m = ModelConfig("test", family, "hf/f", engine="aria", quantizations=["q4_channel"])
        assert m.is_vla is True

    m2 = ModelConfig("test", "qwen", "hf/f", engine="aria", quantizations=["q4_channel"])
    assert m2.is_vla is False
