"""
Embedding 模型封装 — 支持 OpenAI 兼容 API 和本地 sentence-transformers 两种模式。

:class:`OpenAILikeEmbedding`
    继承自 LlamaIndex ``BaseEmbedding``，通过 :func:`rag_module.utils.make_openai_clients`
    获取 ``openai.OpenAI`` / ``openai.AsyncOpenAI`` SDK 实例，调用 vLLM / OpenAI 兼容的
    ``/v1/embeddings`` 端点，支持同步和异步调用，以及私有部署网关的 ``access-key`` 请求头。

:class:`SentenceTransformerEmbedding`
    继承自 LlamaIndex ``BaseEmbedding``，使用 sentence-transformers 在本地推理，
    不依赖外部 Embedding API 服务。

实现参考 FlexRAG ``flexrag/components/retrieval/base.py`` 中的同名类，
与知识库构建阶段使用完全相同的 Embedding 接口，保证向量维度一致。

依赖::

    pip install llama-index openai           # OpenAILikeEmbedding
    pip install llama-index sentence-transformers  # SentenceTransformerEmbedding
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional

from rag_module.utils import first_env, make_openai_clients

# guard：llama-index + pydantic（SentenceTransformerEmbedding 和 OpenAILikeEmbedding
# 都需要，因此单独保护）。
try:
    from pydantic import PrivateAttr
    from llama_index.core.embeddings import BaseEmbedding as _BaseEmbedding
    _LLAMA_AVAILABLE = True
except ImportError:
    _LLAMA_AVAILABLE = False
    _BaseEmbedding = object  # type: ignore[assignment,misc]
    PrivateAttr = lambda **kw: None  # type: ignore[assignment]  # noqa: E731


class OpenAILikeEmbedding(_BaseEmbedding):
    """
    最小化 OpenAI 兼容 Embedding 客户端，继承自 LlamaIndex ``BaseEmbedding``。

    使用 ``openai.OpenAI`` / ``openai.AsyncOpenAI`` SDK 调用 vLLM / OpenAI 兼容的
    ``/v1/embeddings`` 端点，同时支持同步和异步两种调用方式。

    实现参考 ``temp/embedding.py`` 中的 ``EmbeddingClient`` 模式，
    与知识库构建阶段使用完全相同的 Embedding 接口，保证向量维度一致。

    Args:
        base_url:             Embedding 服务的 base URL（如 ``http://127.0.0.1:8018/v1``）。
        model_name:           Embedding 模型名称（如 ``bge-large-zh-v1.5``）。
        api_key:              API Key；本地部署时可设为任意非空字符串（默认 ``"EMPTY"``）。
        access_key:           可选的 ``access-key`` 请求头（部分私有部署网关需要）。
        normalize_embeddings: 是否对返回向量做 L2 归一化（默认 ``True``）。
                              归一化后向量为单位向量，与 ``IndexFlatIP`` 配合时
                              内积等价于余弦相似度，排名不受向量模长干扰。
                              **注意**：构建知识库与检索查询必须使用相同的设置；
                              修改此参数后需重建知识库索引。
    """
    base_url: str
    model_name: str
    api_key: Optional[str] = None
    access_key: Optional[str] = None
    normalize_embeddings: bool = True
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
        normalize_embeddings: bool = True,
        embed_batch_size: int = 5,
        **kwargs: Any,
    ) -> None:
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "OpenAILikeEmbedding 需要安装 llama-index：\n"
                "pip install llama-index"
            )
        super().__init__(
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            access_key=access_key,
            normalize_embeddings=normalize_embeddings,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._client, self._aclient = make_openai_clients(base_url, api_key, access_key)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _maybe_normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        """若 normalize_embeddings=True，对每个向量做 L2 归一化（原地风格，返回新列表）。"""
        if not self.normalize_embeddings:
            return embeddings
        import numpy as np
        vecs = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vecs / norms).tolist()

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
        base_url = first_env(
            ["EMBEDDING_BASE_URL", "QWEN3_BASE_URL", "LLM_BASE_URL"],
            default="http://127.0.0.1:8018/v1",
        )
        api_key = first_env(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"])
        access_key = first_env(
            [
                "EMBEDDING_ACCESS_KEY",
                "QWEN3_ACCESS_KEY",
                "CONTEXTMIND_ACCESS_KEY",
                "LLM_ACCESS_KEY",
                "ACCESS_KEY",
            ]
        )
        model = first_env(
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
        embeddings = [item.embedding for item in response.data]
        return self._maybe_normalize(embeddings)

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
        embeddings = [item.embedding for item in response.data]
        return self._maybe_normalize(embeddings)


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
        # 先检查依赖，避免因 _BaseEmbedding=object 而产生误导性的 TypeError
        if not _LLAMA_AVAILABLE:
            raise ImportError(
                "SentenceTransformerEmbedding 需要安装 llama-index：\n"
                "pip install llama-index sentence-transformers"
            )
        _name = model_name or os.path.basename(model_path.rstrip("/\\")) or "local"
        super().__init__(
            model_path=model_path,
            model_name=_name,
            embed_batch_size=embed_batch_size,
            **kwargs,
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
        # sentence_transformers.encode() 是 CPU 密集型阻塞调用。
        # 直接在协程里调用会阻塞 asyncio 事件循环，在某些 llama-index 版本的
        # span/callback 机制下还会触发 "This event loop is already running"。
        # 正确做法：通过 asyncio.to_thread() 将其推入线程池执行。
        return await asyncio.to_thread(self._get_text_embeddings, texts)

