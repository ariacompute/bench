"""多引擎推理执行模块。

实现引擎适配器模式，将不同推理引擎统一为 run(prompt) → dict 接口。
支持三种引擎:
  - aria:        aria-engine 子进程（Hadamard+Lloyd-Max 量化）
  - llama_cpp:   llama-cli 子进程（GGUF 量化）
  - transformers: Python 脚本子进程（HF pipeline 原生推理）
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 默认超时秒数
DEFAULT_TIMEOUT = 600


def _safe_run(
    cmd: List[str],
    timeout: int = DEFAULT_TIMEOUT,
    env: Optional[Dict] = None,
    cwd: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """安全执行子进程，捕获 stdout/stderr。"""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(  # nosec B603
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=merged_env,
        cwd=cwd,
    )


# ── 抽象适配器 ──


class EngineAdapter(ABC):
    """推理引擎抽象基类。"""

    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout

    @abstractmethod
    def run(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """执行推理。

        Returns:
            dict with keys:
                - text: str, 生成的文本
                - tokens: int, 生成的 token 数
                - finish_reason: str
                - ttft_ms: float (可选)
                - tokens_per_second: float (可选)
                - error: str (若失败)
        """
        ...


# ── Aria Engine 适配器 ──


class AriaEngineAdapter(EngineAdapter):
    """aria-engine 子进程推理。

    命令格式:
        aria-engine run --model <path> --prompt "<prompt>" --max-tokens <n> --temperature <t>
    输出: JSON lines, 每行 {"text": "...", "finish_reason": "...", ...}
    """

    BINARY = "aria-engine"

    def run(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cmd = [
            self.BINARY, "run",
            "--model", model_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temperature", str(temperature),
        ]
        if kwargs.get("seed") is not None:
            cmd += ["--seed", str(kwargs["seed"])]

        try:
            result = _safe_run(cmd, timeout=self.timeout)
            if result.returncode != 0:
                logger.error(
                    "aria-engine 推理失败 (rc=%d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
                return {
                    "text": "", "tokens": 0, "finish_reason": "error",
                    "error": result.stderr[:500],
                }

            output = result.stdout.strip()
            if not output:
                logger.warning("aria-engine 返回空输出")
                return {"text": "", "tokens": 0, "finish_reason": "empty"}

            # 解析 JSON lines，取最后一行作为最终输出
            lines = output.splitlines()
            last = {}
            for line in lines:
                try:
                    last = json.loads(line)
                except json.JSONDecodeError:
                    continue

            text = last.get("text", "")
            tokens = last.get("tokens_generated", len(text.split()))
            return {
                "text": text,
                "tokens": tokens,
                "finish_reason": last.get("finish_reason", "stop"),
                "ttft_ms": last.get("ttft_ms"),
                "tokens_per_second": last.get("tokens_per_second"),
            }
        except subprocess.TimeoutExpired:
            logger.error("aria-engine 推理超时 (%ds)", self.timeout)
            return {
                "text": "", "tokens": 0, "finish_reason": "timeout",
                "error": f"Timeout after {self.timeout}s",
            }
        except Exception as e:
            logger.error("aria-engine 推理异常: %s", e)
            return {"text": "", "tokens": 0, "finish_reason": "error", "error": str(e)}


# ── llama.cpp 适配器 ──


class LlamaCppAdapter(EngineAdapter):
    """llama.cpp llama-cli 子进程推理。

    命令格式:
        llama-cli -m <gguf_path> -p "<prompt>" -n <max_tokens> --temp <t> -e
    输出: stdout 中提取文本行，统计 token 数。
    """

    BINARY = "llama-cli"

    def run(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # 找到实际 GGUF 文件
        actual_path = self._find_gguf(model_path)
        if not actual_path:
            return {
                "text": "", "tokens": 0, "finish_reason": "error",
                "error": f"GGUF 文件未找到: {model_path}",
            }

        cmd = [
            self.BINARY,
            "-m", actual_path,
            "-p", prompt,
            "-n", str(max_tokens),
            "--temp", str(temperature),
            "-e",  # escape sequences
            "--no-display-prompt",
            "--simple-io",
        ]
        if kwargs.get("seed") is not None:
            cmd += ["-s", str(kwargs["seed"])]

        # 额外参数透传
        for param in ("ctx-size", "threads", "batch-size"):
            if kwargs.get(param) is not None:
                cmd += ["--" + param, str(kwargs[param])]

        try:
            result = _safe_run(cmd, timeout=self.timeout)
            if result.returncode != 0:
                logger.error(
                    "llama-cli 推理失败 (rc=%d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
                return {
                    "text": "", "tokens": 0, "finish_reason": "error",
                    "error": result.stderr[:500],
                }

            text = result.stdout.strip()
            # 简单估算 token 数（可用子词统计近似）
            tokens = len(text.split()) if text else 0
            return {
                "text": text,
                "tokens": tokens,
                "finish_reason": "stop",
            }
        except subprocess.TimeoutExpired:
            logger.error("llama-cli 推理超时 (%ds)", self.timeout)
            return {
                "text": "", "tokens": 0, "finish_reason": "timeout",
                "error": f"Timeout after {self.timeout}s",
            }
        except Exception as e:
            logger.error("llama-cli 推理异常: %s", e)
            return {"text": "", "tokens": 0, "finish_reason": "error", "error": str(e)}

    def _find_gguf(self, model_path: str) -> Optional[str]:
        """在 model_path 目录下查找 .gguf 文件。"""
        if os.path.isfile(model_path) and model_path.endswith(".gguf"):
            return model_path
        if not os.path.isdir(model_path):
            return None
        gguf_files = sorted(f for f in os.listdir(model_path) if f.endswith(".gguf"))
        return os.path.join(model_path, gguf_files[0]) if gguf_files else None


# ── Transformers 适配器 ──


TRANSFORMERS_INFERENCE_SCRIPT = r"""
import json, sys, os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
model_dir = sys.argv[1]
prompt = sys.argv[2]
max_tokens = int(sys.argv[3])
temperature = float(sys.argv[4])
quantization = sys.argv[5] if len(sys.argv) > 5 else "fp16"

import torch

# 抑制 verbose 日志
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    load_kwargs = {"torch_dtype": torch.float16, "trust_remote_code": True}
    if quantization == "int8":
        load_kwargs["load_in_8bit"] = True
    model = AutoModelForCausalLM.from_pretrained(model_dir, **load_kwargs)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    tokens = len(generated)
    print(json.dumps({
        "text": text,
        "tokens": tokens,
        "finish_reason": "stop",
        "ttft_ms": None,
        "tokens_per_second": None if elapsed == 0 else tokens / elapsed,
    }))
    sys.exit(0)
except Exception as e:
    print(json.dumps({"text": "", "tokens": 0, "finish_reason": "error", "error": str(e)}))
    sys.exit(1)
"""


class TransformersAdapter(EngineAdapter):
    """HuggingFace Transformers Python 子进程推理。

    通过 Python 子进程加载模型并推理，避免主进程引入 torch 依赖。
    使用内联脚本，无需额外文件。
    """

    PYTHON_BIN = "python3"

    def run(
        self,
        model_path: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        quantization = kwargs.get("quantization", "fp16")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", prefix="bench_infer_", encoding="utf-8", delete=False
        ) as script_file:
            script_file.write(TRANSFORMERS_INFERENCE_SCRIPT)
            script_path = script_file.name

        try:
            cmd = [
                self.PYTHON_BIN, script_path,
                model_path, prompt,
                str(max_tokens), str(temperature), str(quantization),
            ]
            result = _safe_run(cmd, timeout=self.timeout)

            if result.returncode != 0:
                # 尝试从 stdout 解析 JSON 错误
                try:
                    err_data = json.loads(result.stdout.strip())
                    return err_data
                except (json.JSONDecodeError, ValueError):
                    pass
                logger.error(
                    "transformers 推理失败 (rc=%d): %s",
                    result.returncode,
                    result.stderr[:500],
                )
                return {
                    "text": "", "tokens": 0, "finish_reason": "error",
                    "error": result.stderr[:500],
                }

            try:
                data = json.loads(result.stdout.strip())
                return data
            except json.JSONDecodeError:
                logger.error("transformers 输出解析失败: %s", result.stdout[:200])
                return {
                    "text": "", "tokens": 0, "finish_reason": "error",
                    "error": "JSON decode failed",
                }
        except subprocess.TimeoutExpired:
            logger.error("transformers 推理超时 (%ds)", self.timeout)
            return {
                "text": "", "tokens": 0, "finish_reason": "timeout",
                "error": f"Timeout after {self.timeout}s",
            }
        except Exception as e:
            logger.error("transformers 推理异常: %s", e)
            return {"text": "", "tokens": 0, "finish_reason": "error", "error": str(e)}
        finally:
            try:
                os.unlink(script_path)
            except OSError:
                pass


# ── 适配器注册表 ──

ADAPTER_REGISTRY: Dict[str, type] = {
    "aria": AriaEngineAdapter,
    "llama_cpp": LlamaCppAdapter,
    "transformers": TransformersAdapter,
}


# ── 调度器 ──


def _build_adapter(engine_type: str, timeout: int = DEFAULT_TIMEOUT) -> EngineAdapter:
    """创建对应引擎的适配器实例。"""
    adapter_cls = ADAPTER_REGISTRY.get(engine_type)
    if adapter_cls is None:
        raise ValueError(f"不支持的引擎类型: {engine_type}. 可用: {list(ADAPTER_REGISTRY)}")
    return adapter_cls(timeout=timeout)


def run_inference(
    model_path: str,
    engine_type: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> Dict[str, Any]:
    """统一推理接口，根据 engine_type 调度到对应适配器。

    Args:
        model_path: 模型本地路径（目录或 GGUF 文件路径）
        engine_type: 引擎类型 (aria / llama_cpp / transformers)
        prompt: 输入 prompt 文本
        max_tokens: 最大生成 token 数
        temperature: 采样温度 (0 = 贪婪)
        timeout: 超时秒数
        **kwargs: 引擎特定参数
            - aria: seed
            - llama_cpp: seed, ctx_size, threads, batch_size, quantization
            - transformers: quantization

    Returns:
        {"text": str, "tokens": int, "finish_reason": str, ...}
    """
    adapter = _build_adapter(engine_type, timeout=timeout)
    return adapter.run(
        model_path=model_path,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs,
    )
