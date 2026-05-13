"""
LLM 客户端 — 基于 LiteLLM，接入本地 vLLM / OpenAI 兼容端点。

实际实现位于 :mod:`model_client.api.llm`，此处重新导出以保持向后兼容性。

Usage::

    from rag_module.llm_client import LLMClient, LLMConfig
    # 或直接从独立模块导入：
    from model_client.api import LLMClient, LLMConfig
"""
from model_client.api.llm import (
    LLMConfig,
    LLMClient,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
)

__all__ = ["LLMConfig", "LLMClient", "DEFAULT_MODEL", "DEFAULT_TEMPERATURE", "DEFAULT_MAX_TOKENS"]
