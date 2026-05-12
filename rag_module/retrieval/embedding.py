"""
OpenAILikeEmbedding — OpenAI 兼容 Embedding 客户端。

继承自 LlamaIndex ``BaseEmbedding``，通过 ``openai.OpenAI`` /
``openai.AsyncOpenAI`` SDK 调用 vLLM / OpenAI 兼容的 ``/v1/embeddings`` 端点，
支持同步和异步两种调用方式，以及私有部署网关的 ``access-key`` 请求头。

实现参考 FlexRAG ``flexrag/components/retrieval/base.py`` 中的同名类，
与知识库构建阶段使用完全相同的 Embedding 接口，保证向量维度一致。

依赖::

    pip install llama-index openai
"""
from __future__ import annotations

from typing import Any, List, Optional

# Guard llama-index / openai import so the whole rag_module package remains importable
# even when these are not installed. Runtime errors are raised in _get_text_embeddings().
try:
    from pydantic import PrivateAttr
    from openai import OpenAI as _OpenAI, AsyncOpenAI as _AsyncOpenAI
    from llama_index.core.embeddings import BaseEmbedding as _BaseEmbedding
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    _BaseEmbedding = object  # type: ignore[assignment,misc]
    PrivateAttr = lambda **kw: None  # type: ignore[assignment]  # noqa: E731
    _OpenAI = None  # type: ignore[assignment]
    _AsyncOpenAI = None  # type: ignore[assignment]


class OpenAILikeEmbedding(_BaseEmbedding):
    """
    最小化 OpenAI 兼容 Embedding 客户端，继承自 LlamaIndex ``BaseEmbedding``。

    使用 ``openai.OpenAI`` / ``openai.AsyncOpenAI`` SDK 调用 vLLM / OpenAI 兼容的
    ``/v1/embeddings`` 端点，同时支持同步和异步两种调用方式。

    实现参考 ``temp/embedding.py`` 中的 ``EmbeddingClient`` 模式，
    与知识库构建阶段使用完全相同的 Embedding 接口，保证向量维度一致。

    Args:
        base_url:    Embedding 服务的 base URL（如 ``http://127.0.0.1:8018/v1``）。
        model_name:  Embedding 模型名称（如 ``bge-large-zh-v1.5``）。
        api_key:     API Key；本地部署时可设为任意非空字符串（默认 ``"EMPTY"``）。
        access_key:  可选的 ``access-key`` 请求头（部分私有部署网关需要）。
    """

    # Pydantic 字段
    base_url: str
    model_name: str
    api_key: Optional[str] = None
    access_key: Optional[str] = None
    embed_batch_size: int = 5  # inherited from BaseEmbedding; declared here for Pydantic

    # 非序列化内部变量
    _client: Any = PrivateAttr()
    _aclient: Any = PrivateAttr()

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: Optional[str] = None,
        access_key: Optional[str] = None,
        embed_batch_size: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            access_key=access_key,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        headers = {}
        if access_key:
            headers["access-key"] = access_key

        self._client = _OpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            default_headers=headers if headers else None,
        )
        self._aclient = _AsyncOpenAI(
            api_key=api_key or "EMPTY",
            base_url=base_url,
            default_headers=headers if headers else None,
        )

    # ------------------------------------------------------------------
    # 同步方法
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]

    # ------------------------------------------------------------------
    # 异步方法
    # ------------------------------------------------------------------

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aget_text_embeddings([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aget_text_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = await self._aclient.embeddings.create(
            model=self.model_name,
            input=texts,
        )
        return [item.embedding for item in response.data]
