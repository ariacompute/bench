"""eval 模块单元测试"""

from unittest.mock import MagicMock, patch

import pytest

from src.eval.academic import _check_answer, _check_code_pass, _load_dataset
from src.eval.engine_runner import (
    AriaEngineAdapter,
    EngineAdapter,
    LlamaCppAdapter,
    TransformersAdapter,
    run_inference,
)
from src.eval.instruction import _check_constraint
from src.eval.longcontext import _generate_filler_text, _insert_needle

# ── engine_runner: 适配器注册 ──


def test_adapter_registry():
    """注册表包含三种引擎"""
    from src.eval.engine_runner import ADAPTER_REGISTRY

    assert "aria" in ADAPTER_REGISTRY
    assert "llama_cpp" in ADAPTER_REGISTRY
    assert "transformers" in ADAPTER_REGISTRY
    assert issubclass(ADAPTER_REGISTRY["aria"], EngineAdapter)
    assert issubclass(ADAPTER_REGISTRY["llama_cpp"], EngineAdapter)
    assert issubclass(ADAPTER_REGISTRY["transformers"], EngineAdapter)


def test_run_inference_unknown_engine():
    """未知引擎应抛出 ValueError"""
    with pytest.raises(ValueError, match="不支持的引擎类型"):
        run_inference("/fake/path", "unknown_engine", prompt="hello")


# ── AriaEngineAdapter ──


@patch("src.eval.engine_runner._safe_run")
def test_aria_adapter_success(mock_safe_run):
    import json

    output_data = {"text": "hello world", "tokens_generated": 5, "finish_reason": "stop", "ttft_ms": 100.0}
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps(output_data)
    mock_safe_run.return_value = mock_proc

    adapter = AriaEngineAdapter(timeout=600)
    result = adapter.run("/fake/bundle", "hello", max_tokens=10, temperature=0.0)
    assert result["text"] == "hello world"
    assert result["tokens"] == 5
    assert result["finish_reason"] == "stop"
    assert "error" not in result


@patch("src.eval.engine_runner._safe_run")
def test_aria_adapter_error(mock_safe_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "engine error: missing model"
    mock_safe_run.return_value = mock_proc

    adapter = AriaEngineAdapter(timeout=600)
    result = adapter.run("/fake/bundle", "hello")
    assert result["finish_reason"] == "error"
    assert "missing model" in result["error"]


# ── LlamaCppAdapter ──


@patch("src.eval.engine_runner._safe_run")
@patch.object(LlamaCppAdapter, "_find_gguf", return_value="/fake/model.gguf")
def test_llama_cpp_adapter_success(mock_find, mock_safe_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = "This is generated text."
    mock_safe_run.return_value = mock_proc

    adapter = LlamaCppAdapter(timeout=600)
    result = adapter.run("/fake/model", "hello", max_tokens=10, temperature=0.0)
    assert result["text"] == "This is generated text."
    assert result["finish_reason"] == "stop"


@patch.object(LlamaCppAdapter, "_find_gguf", return_value=None)
def test_llama_cpp_no_gguf(mock_find):
    adapter = LlamaCppAdapter(timeout=600)
    result = adapter.run("/fake/missing", "hello")
    assert result["finish_reason"] == "error"
    assert "GGUF" in result["error"]


# ── TransformersAdapter ──


@patch("src.eval.engine_runner._safe_run")
def test_transformers_adapter_success(mock_safe_run):
    import json

    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({"text": "generated", "tokens": 8, "finish_reason": "stop"})
    mock_safe_run.return_value = mock_proc

    adapter = TransformersAdapter(timeout=600)
    result = adapter.run("/fake/model", "hello", max_tokens=10, temperature=0.0)
    assert result["text"] == "generated"
    assert result["finish_reason"] == "stop"


@patch("src.eval.engine_runner._safe_run")
def test_transformers_adapter_error(mock_safe_run):
    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "CUDA out of memory"
    mock_safe_run.return_value = mock_proc

    adapter = TransformersAdapter(timeout=600)
    result = adapter.run("/fake/model", "hello")
    assert result["finish_reason"] == "error"


# ── 统一接口 run_inference ──


@patch.object(AriaEngineAdapter, "run")
def test_run_inference_dispatches_to_aria(mock_run):
    mock_run.return_value = {"text": "hi", "tokens": 1, "finish_reason": "stop"}
    result = run_inference("/fake", "aria", "hello")
    assert result["text"] == "hi"
    mock_run.assert_called_once()


@patch.object(LlamaCppAdapter, "run")
def test_run_inference_dispatches_to_llama_cpp(mock_run):
    mock_run.return_value = {"text": "hi", "tokens": 1, "finish_reason": "stop"}
    result = run_inference("/fake.gguf", "llama_cpp", "hello")
    assert result["text"] == "hi"


@patch.object(TransformersAdapter, "run")
def test_run_inference_dispatches_to_transformers(mock_run):
    mock_run.return_value = {"text": "hi", "tokens": 1, "finish_reason": "stop"}
    result = run_inference("/fake", "transformers", "hello")
    assert result["text"] == "hi"


# ── academic ──


def test_load_sample_dataset():
    data = _load_dataset("mmlu")
    assert len(data) == 2
    assert "question" in data[0]
    assert "answer" in data[0]


def test_check_answer_mmlu():
    assert _check_answer("B. Paris", "B", "mmlu") is True
    assert _check_answer("The answer is A", "C", "mmlu") is False


def test_check_answer_gsm8k():
    assert _check_answer("The final answer is 8.", "8", "gsm8k") is True


def test_check_code_pass():
    assert _check_code_pass("def foo():\n    return True", "def foo():\n    return True") is True
    assert _check_code_pass("", "def foo():\n    return True") is False


# ── longcontext ──


def test_generate_filler_text():
    text = _generate_filler_text(4096)
    assert len(text) > 0


def test_insert_needle():
    text = " ".join(["word"] * 100)
    needle = "SECRET"
    result = _insert_needle(text, needle, 50)
    assert needle in result
    assert len(result) > len(text)


# ── instruction ──


def test_check_constraint_sentence_count():
    assert _check_constraint("A. B. C.", "sentence_count=3") is True
    assert _check_constraint("Only one sentence.", "sentence_count=3") is False


def test_check_constraint_contains():
    assert _check_constraint("I love machine learning.", "contains=machine") is True
    assert _check_constraint("I love deep learning.", "contains=machine") is False


def test_check_constraint_startswith():
    assert _check_constraint("Artificial intelligence is amazing.", "startswith=Artificial") is True
    assert _check_constraint("Hello world.", "startswith=Artificial") is False
