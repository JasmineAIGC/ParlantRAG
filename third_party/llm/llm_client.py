"""基于 Google ADK 的 LLM 客户端，通过 LiteLlm 接入本地 vLLM/OpenAI 兼容端点。"""

from __future__ import annotations

import asyncio
import os
import re
import threading
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

# ---------------------------------------------------------------------------
# 全局共享事件循环（运行在独立后台守护线程中）
# ---------------------------------------------------------------------------

_shared_loop_lock = threading.Lock()
_shared_loop: Optional[asyncio.AbstractEventLoop] = None


def _shared_loop_exception_handler(
    loop: asyncio.AbstractEventLoop,
    context: dict,
) -> None:
    """抑制 litellm 异步日志任务 GC 时的噪音，其余异常交由默认处理器处理。"""
    msg = context.get("message", "")
    if "Task was destroyed but it is pending" in msg:
        return
    loop.default_exception_handler(context)


def _get_shared_loop() -> asyncio.AbstractEventLoop:
    """返回全局唯一共享事件循环（永久运行在后台守护线程中）。"""
    global _shared_loop
    if _shared_loop is None or _shared_loop.is_closed():
        with _shared_loop_lock:
            if _shared_loop is None or _shared_loop.is_closed():
                loop = asyncio.new_event_loop()
                loop.set_exception_handler(_shared_loop_exception_handler)
                t = threading.Thread(
                    target=loop.run_forever,
                    name="contextmind-llm-event-loop",
                    daemon=True,
                )
                t.start()
                _shared_loop = loop
    return _shared_loop

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google.adk.agents import LlmAgent  # type: ignore[import]  # noqa: F401 — kept for create_local_adk_model / external callers
from google.adk.models.lite_llm import LiteLlm  # type: ignore[import]
from google.adk.runners import InMemoryRunner  # type: ignore[import]  # noqa: F401 — kept for external callers

from .logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# LiteLLM 噪音抑制：屏蔽 async_success_handler RuntimeWarning，
# 并注册零价格条目以消除本地模型"not in cost map"的 DEBUG 日志。
# ---------------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="coroutine 'Logging.async_success_handler' was never awaited",
    category=RuntimeWarning,
)

def _configure_litellm_for_local_model(model_name: str) -> None:
    """配置 LiteLLM 全局状态：禁用回调、注册零价格条目、抑制噪音日志。"""
    # 将 LiteLLM logger 级别提升至 WARNING，屏蔽每次调用的 INFO 日志
    import logging as _logging
    _litellm_logger = _logging.getLogger("LiteLLM")
    _litellm_logger.propagate = False
    _litellm_logger.setLevel(_logging.WARNING)

    try:
        import litellm  # type: ignore[import]

        litellm.success_callback = []
        litellm.failure_callback = []
        litellm.callbacks = []
        # 抑制 litellm 在 stdout 输出 "Provider List:" 等调试信息
        litellm.suppress_debug_info = True

        # 注册零价格条目，覆盖 `openai/<model>` 和裸 model 两种键。
        _zero_cost: dict = {
            "max_tokens": 32768,
            "max_input_tokens": 32768,
            "max_output_tokens": 32768,
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "litellm_provider": "openai",
            "mode": "chat",
        }
        model_registry = {key: _zero_cost for key in (model_name, f"openai/{model_name}")}
        try:
            litellm.register_model(model_registry)
        except Exception:
            # 直接写入 model_cost dict（兼容旧版 litellm）
            for key, val in model_registry.items():
                litellm.model_cost.setdefault(key, val)
    except Exception as exc:
        logger.debug(f"[AdkLLMClient] litellm 配置失败（非致命）: {exc}")


DEFAULT_MODEL = "Qwen3-32B"
DEFAULT_TEMPERATURE = 0.3


def _first_env(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """按顺序读取环境变量，返回首个非空值。"""
    for key in keys:
        if value := os.getenv(key):
            return value
    return default


_THINKING_MODELS = ("qwen3", "qwen-3", "deepseek-r1", "deepseek_r1")
"""需要禁用思考模式的推理模型名称片段（小写）。"""

_LITELLM_PROVIDER_PREFIXES = (
    "openai/", "anthropic/", "azure/", "gemini/", "google/",
    "mistral/", "cohere/", "huggingface/", "deepseek/", "ollama/",
    "bedrock/", "vertex_ai/", "groq/", "together_ai/", "replicate/",
    "aleph_alpha/", "baseten/", "palm/", "nlp_cloud/", "petals/",
    "voyage/", "cloudflare/", "xinference/", "fireworks_ai/",
)
"""LiteLlm 已知的提供商前缀，用于判断模型名是否已含提供商信息。"""


def _is_thinking_model(model: str) -> bool:
    """如果模型名称表明它是推理型模型，返回 True。"""
    lower = model.lower()
    return any(fragment in lower for fragment in _THINKING_MODELS)


def _needs_openai_prefix(model: str) -> bool:
    """判断模型名是否需要添加 ``openai/`` 前缀。

    LiteLlm 使用 ``provider/model`` 格式路由请求。若模型名称已包含已知提供商前缀
    （如 ``anthropic/``、``gemini/`` 等），则无需追加 ``openai/`` 前缀；否则默认
    将其视为 OpenAI 兼容端点，添加 ``openai/`` 前缀以正确路由。
    """
    return not any(model.startswith(p) for p in _LITELLM_PROVIDER_PREFIXES)


def _build_thinking_extra_body(
    resolved_thinking: Optional[bool],
    base_url: Optional[str],
) -> dict:
    """构建思考模式控制参数的 ``extra_body`` 字典。

    有 base_url（vLLM）时嵌套在 ``chat_template_kwargs`` 下；无 base_url（云端 API）时放顶层。
    ``resolved_thinking`` 为 ``None`` 时返回空字典。
    """
    if resolved_thinking is None:
        return {}
    if base_url:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": resolved_thinking}}}
    else:
        return {"extra_body": {"enable_thinking": resolved_thinking}}


@dataclass
class LLMConfig:
    """OpenAI 兼容端点的运行时配置。"""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: Optional[str] = None
    temperature: float = DEFAULT_TEMPERATURE
    top_k: int = 20
    max_tokens: int = 4096
    enable_thinking: Optional[bool] = None
    """控制推理模型（如 Qwen3）的思考模式。
    ``False`` 禁用链式推理输出；``None`` 对已知推理模型自动设为 ``False``；``True`` 强制启用。
    """

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """从环境变量加载配置。"""
        api_key = _first_env([
            "QWEN3_API_KEY", "CONTEXTMIND_API_KEY", "LLM_API_KEY",
            "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "QWEN_API_KEY",
        ])
        if not api_key:
            raise ValueError(
                "缺少 API key。请在 .env 文件中设置以下变量之一: "
                "QWEN3_API_KEY, LLM_API_KEY, OPENAI_API_KEY 等"
            )

        base_url = _first_env([
            "QWEN3_BASE_URL", "CONTEXTMIND_BASE_URL", "LLM_BASE_URL",
            "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", "QWEN_BASE_URL",
        ])
        model = _first_env([
            "QWEN3_MODEL", "CONTEXTMIND_MODEL", "LLM_MODEL",
            "OPENAI_MODEL", "DEEPSEEK_MODEL", "QWEN_MODEL",
        ], default=DEFAULT_MODEL)
        temperature_str = _first_env([
            "CONTEXTMIND_TEMPERATURE", "LLM_TEMPERATURE", "OPENAI_TEMPERATURE",
        ])
        temperature = float(temperature_str) if temperature_str else DEFAULT_TEMPERATURE
        top_k = int(v) if (v := _first_env(["QWEN3_TOP_K", "LLM_TOP_K", "TOP_K"])) else 20
        max_tokens = int(v) if (v := _first_env(["QWEN3_MAX_TOKENS", "LLM_MAX_TOKENS", "MAX_TOKENS"])) else 4096

        # 解析 ENABLE_THINKING 环境变量；若未设置则由 enable_thinking=None 触发自动检测
        enable_thinking_str = _first_env([
            "CONTEXTMIND_ENABLE_THINKING", "LLM_ENABLE_THINKING", "ENABLE_THINKING",
        ])
        if enable_thinking_str is not None:
            enable_thinking: Optional[bool] = enable_thinking_str.lower() not in ("false", "0", "no", "off")
        else:
            enable_thinking = None

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            top_k=top_k,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        )


def create_local_adk_model(config: Optional[LLMConfig] = None) -> Any:
    """从 LLMConfig 创建 ADK LiteLlm 模型实例，接入本地 vLLM/OpenAI 兼容端点。

    示例::

        from contextmind.infra.llm_client import create_local_adk_model
        from google.adk.agents import LlmAgent

        model = create_local_adk_model()
        agent = LlmAgent(model=model, name="my_agent", instruction="你是助手。")
    """
    if config is None:
        config = LLMConfig.from_env()

    # None → 对已知推理模型自动禁用思考；False → 强制禁用；True → 强制启用
    resolved_thinking: Optional[bool] = config.enable_thinking
    if resolved_thinking is None and _is_thinking_model(config.model):
        resolved_thinking = False

    extra_kwargs = _build_thinking_extra_body(resolved_thinking, config.base_url)

    # 若模型名称已含提供商前缀，则不再追加 openai/ 前缀
    model_str = f"openai/{config.model}" if _needs_openai_prefix(config.model) else config.model

    return LiteLlm(
        model=model_str,
        api_base=config.base_url,
        api_key=config.api_key,
        **extra_kwargs,
    )


class AdkLLMClient:
    """基于 Google ADK（LiteLLM）的无状态 LLM 客户端。

    使用 ``litellm.acompletion()`` 直接调用 vLLM / OpenAI 兼容端点，
    每次 ``chat_async`` 调用恰好发起一次 LLM API 请求（无 Agent 框架开销）。

    参数
    ----
    config : LLMConfig | None
        本地 LLM 配置；若为 None 则从 ``.env`` 自动加载。

    示例::

        from contextmind.infra.llm_client import AdkLLMClient
        llm = AdkLLMClient()
        reply = await llm.chat_async([{"role": "user", "content": "你好"}])
    """

    def __init__(
        self,
        system_instruction: str = "你是一个智能对话助手。",  # noqa: ARG002 — 保持旧接口兼容
        config: Optional[LLMConfig] = None,
    ) -> None:
        # fail-fast：在初始化时导入 litellm，避免在 chat_async 时才发现依赖缺失
        try:
            import litellm as _litellm  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "litellm 未安装，无法调用 LLM。\n请运行: pip install litellm"
            ) from exc
        self._litellm = _litellm

        self._config = config or LLMConfig.from_env()
        _configure_litellm_for_local_model(self._config.model)

        resolved_thinking: Optional[bool] = self._config.enable_thinking
        if resolved_thinking is None and _is_thinking_model(self._config.model):
            resolved_thinking = False
        self._resolved_thinking = resolved_thinking

        logger.info(f"[AdkLLMClient] 初始化完成，model={self._config.model}")

    async def chat_async(
        self,
        messages: List[dict],
        temperature: Optional[float] = None,
    ) -> str:
        """异步多轮对话。每次调用恰好发起一次 LLM API 请求。

        Args:
            messages: OpenAI 格式消息列表（role/content 字典）。
            temperature: 覆盖默认温度；为 None 时使用 LLMConfig 中的值。

        Returns:
            助手的回复文本（已剥除推理模型的 <think>…</think> 块）。
        """
        _litellm = self._litellm
        temp = temperature if temperature is not None else self._config.temperature

        # 根据部署方式选择正确的 enable_thinking 参数格式
        extra_kwargs = _build_thinking_extra_body(self._resolved_thinking, self._config.base_url)

        model_str = f"openai/{self._config.model}" if _needs_openai_prefix(self._config.model) else self._config.model

        response = await _litellm.acompletion(
            model=model_str,
            messages=messages,
            api_base=self._config.base_url,
            api_key=self._config.api_key,
            temperature=temp,
            max_tokens=self._config.max_tokens,
            **extra_kwargs,
        )
        final_text: str = (response.choices[0].message.content or "").strip()
        # 剥除推理模型（Qwen3、DeepSeek-R1 等）输出的思考块
        final_text = re.sub(r"<think>.*?</think>", "", final_text, flags=re.DOTALL)
        return final_text.strip()

    async def generate_async(self, prompt: str) -> tuple[str, str]:
        """异步 prompt 生成，返回 ``("", response_text)``。"""
        return "", await self.chat_async([{"role": "user", "content": prompt}])

    def generate(self, prompt: str) -> tuple[str, str]:
        """同步 prompt 生成，返回 ``(reasoning, response_text)`` 元组。

        若已有运行中的 event loop（如 Jupyter），在独立线程中执行以避免嵌套 loop 报错。
        否则通过全局共享事件循环执行，确保多线程并发时 litellm 绑定到同一 loop。
        """
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, self.generate_async(prompt))
                return future.result()
        else:
            fut = asyncio.run_coroutine_threadsafe(
                self.generate_async(prompt), _get_shared_loop()
            )
            return fut.result()

