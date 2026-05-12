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

import os
from typing import Any, List, Optional, Sequence

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


def _first_env(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """按顺序读取环境变量，返回首个非空值。"""
    for key in keys:
        if value := os.getenv(key):
            return value
    return default


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
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "OpenAILikeEmbedding":
        """
        从环境变量（或 ``.env`` 文件）创建实例，与 ``FlexRAGRetriever.from_env()``
        使用完全相同的环境变量命名和优先级顺序：

        ================  ==========================================
        参数               读取的环境变量
        ================  ==========================================
        base_url          EMBEDDING_BASE_URL / QWEN3_BASE_URL / LLM_BASE_URL
        api_key           EMBEDDING_API_KEY / QWEN3_API_KEY / LLM_API_KEY
        access_key        EMBEDDING_ACCESS_KEY / QWEN3_ACCESS_KEY /
                          CONTEXTMIND_ACCESS_KEY / LLM_ACCESS_KEY / ACCESS_KEY
        model_name        EMBEDDING_MODEL / QWEN3_MODEL / LLM_MODEL
        ================  ==========================================
        """
        base_url = _first_env(
            ["EMBEDDING_BASE_URL", "QWEN3_BASE_URL", "LLM_BASE_URL"],
            default="http://127.0.0.1:8018/v1",
        )
        api_key = _first_env(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"])
        access_key = _first_env(
            [
                "EMBEDDING_ACCESS_KEY",
                "QWEN3_ACCESS_KEY",
                "CONTEXTMIND_ACCESS_KEY",
                "LLM_ACCESS_KEY",
                "ACCESS_KEY",
            ]
        )
        model = _first_env(
            ["EMBEDDING_MODEL", "QWEN3_MODEL", "LLM_MODEL"],
            default="bge-large-zh-v1.5",
        )
        return cls(base_url=base_url, model_name=model, api_key=api_key, access_key=access_key)

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


class SentenceTransformerEmbedding(_BaseEmbedding):
    """
    LlamaIndex 兼容的本地 Embedding 模型封装，使用 sentence-transformers 库在本地推理。

    适用于不依赖外部 Embedding API 服务、直接加载本地模型文件的场景。
    与 ``OpenAILikeEmbedding`` 接口完全一致，可无缝替换传入 ``FlexRAGRetriever``。

    依赖::

        pip install sentence-transformers

    Args:
        model_path:  本地模型目录路径（传给 ``SentenceTransformer()``），
                     也可以是 HuggingFace Hub 模型名称（如 ``BAAI/bge-large-zh-v1.5``）。
        model_name:  可选的可读模型名称（用于日志/属性展示），默认取 ``model_path`` 的最后一段。
    """

    model_path: str
    model_name: str = "local"
    embed_batch_size: int = 32

    _model: Any = PrivateAttr()

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        embed_batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        _name = model_name or os.path.basename(model_path.rstrip("/\\")) or "local"
        super().__init__(
            model_path=model_path,
            model_name=_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "SentenceTransformerEmbedding 需要安装 llama-index 和 sentence-transformers：\n"
                "pip install llama-index sentence-transformers"
            )
        try:
            from sentence_transformers import SentenceTransformer as _ST
        except ImportError:
            raise ImportError(
                "SentenceTransformerEmbedding 需要安装 sentence-transformers：\n"
                "pip install sentence-transformers"
            )
        self._model = _ST(model_path)

    @classmethod
    def from_env(cls) -> "SentenceTransformerEmbedding":
        """从 ``EMBEDDING_MODEL_PATH`` 环境变量创建实例。"""
        model_path = os.getenv("EMBEDDING_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "EMBEDDING_MODEL_PATH 未设置，无法使用本地 embedding 模式。"
            )
        model_name = os.getenv("EMBEDDING_MODEL") or None
        return cls(model_path=model_path, model_name=model_name)

    # ------------------------------------------------------------------
    # 同步方法
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embeddings([query])[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embeddings([text])[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        import numpy as np
        vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs).tolist()

    # ------------------------------------------------------------------
    # 异步方法
    # ------------------------------------------------------------------

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return (await self._aget_text_embeddings([query]))[0]

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return (await self._aget_text_embeddings([text]))[0]

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        import asyncio
        return await asyncio.get_running_loop().run_in_executor(
            None, self._get_text_embeddings, texts
        )

