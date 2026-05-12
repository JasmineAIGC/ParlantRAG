"""
FlexRAGRetriever — 加载 FlexRAG 构建的 FAISS 向量知识库。

知识库调用方式参考 https://github.com/Xavier-123/FlexRAG 项目中
``MultiVectorRetriever`` 和 ``OpenAILikeEmbedding`` 的实现（因为知识库本身就是由
该项目的 ``scripts/build_knowledge_base.py`` 构建的，使用完全一致的索引格式与调用
模式，确保 Embedding 和 FAISS 索引的兼容性）。

主要组件对应关系：

=============  ===============================================
本模块            FlexRAG 原始位置
=============  ===============================================
OpenAILikeEmbedding   flexrag/components/retrieval/base.py
FlexRAGRetriever      flexrag/components/retrieval/multi_vector_retriever.py
                      (MultiVectorRetriever._load_index + retrieve)
=============  ===============================================

依赖::

    pip install llama-index llama-index-vector-stores-faiss faiss-cpu httpx
    # BM25 混合（可选）：
    pip install llama-index-retrievers-bm25 rank_bm25 jieba
"""
from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any, List, Optional, Sequence

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

# Guard llama-index / openai import so the whole rag_module package remains importable
# even when these are not installed. Runtime errors are raised in _load_index().
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

from rag_module.base import BaseRetriever, Chunk, RetrievalResult

_FAISS_FILE = "faiss_index.bin"


def _first_env_local(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """按顺序读取环境变量，返回首个非空值。"""
    for key in keys:
        if value := os.getenv(key):
            return value
    return default


# ---------------------------------------------------------------------------
# OpenAILikeEmbedding
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FlexRAGRetriever
# ---------------------------------------------------------------------------

class FlexRAGRetriever(BaseRetriever):
    """
    读取由 FlexRAG ``MultiVectorRetriever`` 持久化的目录，执行向量（+可选 BM25）检索。

    索引加载方式与 FlexRAG ``MultiVectorRetriever._load_index()`` 完全一致：

    1. 读取 ``{persist_dir}/faiss_index.bin``
    2. 构造 ``FaissVectorStore`` + ``StorageContext``
    3. 调用 ``load_index_from_storage()`` 恢复 ``VectorStoreIndex``
    4. 通过 ``index.as_retriever()`` 执行相似度检索

    构造时立即加载索引（同步），检索时异步执行。

    Usage::

        retriever = FlexRAGRetriever(
            persist_dir="./knowledge_base",
            embedding_base_url="http://127.0.0.1:8018/v1",
            embedding_model="bge-large-zh-v1.5",
            top_k=5,
        )
        # Or load everything from .env / environment variables:
        retriever = FlexRAGRetriever.from_env()
        result = await retriever.retrieve("查询词")
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_base_url: str = "http://127.0.0.1:8018/v1",
        embedding_api_key: Optional[str] = None,
        embedding_access_key: Optional[str] = None,
        embedding_model: str = "bge-large-zh-v1.5",
        top_k: int = 5,
        bm25_weight: float = 0.0,
    ) -> None:
        """
        Args:
            persist_dir:            FlexRAG 知识库根目录（包含 ``faiss_index.bin`` 等文件）。
            embedding_base_url:     Embedding 服务的 base URL（vLLM / OpenAI 兼容）。
            embedding_api_key:      Embedding 服务的 API Key；本地部署时可省略或设为任意字符串。
            embedding_access_key:   可选的 ``access-key`` 请求头（部分私有部署网关需要）。
            embedding_model:        Embedding 模型名称，须与构建知识库时完全一致。
            top_k:                  默认返回的片段数量。
            bm25_weight:            BM25 混合权重（0~1）。0.0 表示纯向量检索。
                                    当 ``persist_dir/bm25_index/`` 存在且此值 > 0 时
                                    自动启用混合检索。
        """
        self._persist_dir = persist_dir
        self._embedding_base_url = embedding_base_url
        self._embedding_api_key = embedding_api_key
        self._embedding_access_key = embedding_access_key
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._bm25_weight = bm25_weight

        self._index: Any = None
        self._embed_model: Any = None
        self._bm25_retriever: Any = None

        self._load_index()

    @classmethod
    def from_env(cls, persist_dir: Optional[str] = None) -> "FlexRAGRetriever":
        """
        从环境变量（或 ``.env`` 文件）加载 Embedding 配置并创建检索器实例。

        读取顺序（优先级高 → 低）：

        ================  ==========================================
        参数               读取的环境变量
        ================  ==========================================
        persist_dir          KNOWLEDGE_PERSIST_DIR / KNOWLEDGE_BASE_DIR
        embedding_base_url   EMBEDDING_BASE_URL / QWEN3_BASE_URL / LLM_BASE_URL
        embedding_api_key    EMBEDDING_API_KEY / QWEN3_API_KEY / LLM_API_KEY
        embedding_access_key EMBEDDING_ACCESS_KEY / QWEN3_ACCESS_KEY /
                             CONTEXTMIND_ACCESS_KEY / LLM_ACCESS_KEY / ACCESS_KEY
        embedding_model      EMBEDDING_MODEL / QWEN3_MODEL / LLM_MODEL
        top_k                TOP_K_RETRIEVAL / TOP_K
        bm25_weight          BM25_WEIGHT
        ====================  ==========================================

        与 ``temp/embedding.py`` 中的 ``EmbeddingConfig.from_env()`` 命名完全兼容。
        """
        pd = persist_dir or _first_env_local(
            ["KNOWLEDGE_PERSIST_DIR", "KNOWLEDGE_BASE_DIR"],
            default="./knowledge_base/knowledge_persist_dir/hotpotqa",
        )
        base_url = _first_env_local(
            ["EMBEDDING_BASE_URL", "QWEN3_BASE_URL", "LLM_BASE_URL"],
            default="http://127.0.0.1:8018/v1",
        )
        api_key = _first_env_local(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"])
        access_key = _first_env_local(
            [
                "EMBEDDING_ACCESS_KEY",
                "QWEN3_ACCESS_KEY",
                "CONTEXTMIND_ACCESS_KEY",
                "LLM_ACCESS_KEY",
                "ACCESS_KEY",
            ]
        )
        model = _first_env_local(
            ["EMBEDDING_MODEL", "QWEN3_MODEL", "LLM_MODEL"],
            default="bge-large-zh-v1.5",
        )
        top_k = int(_first_env_local(["TOP_K_RETRIEVAL", "TOP_K"]) or "5")
        bm25_weight = float(_first_env_local(["BM25_WEIGHT"]) or "0.0")

        return cls(
            persist_dir=pd,
            embedding_base_url=base_url,
            embedding_api_key=api_key,
            embedding_access_key=access_key,
            embedding_model=model,
            top_k=top_k,
            bm25_weight=bm25_weight,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def persist_dir(self) -> str:
        """知识库根目录路径。"""
        return self._persist_dir

    @property
    def embedding_model(self) -> str:
        """Embedding 模型名称。"""
        return self._embedding_model

    @property
    def embedding_base_url(self) -> str:
        """Embedding 服务的 base URL。"""
        return self._embedding_base_url

    @property
    def doc_count(self) -> int:
        """已加载的文档片段总数；索引未加载时返回 0。"""
        if self._index is None:
            return 0
        try:
            return len(self._index.docstore.docs)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """
        加载 FAISS 持久化索引（以及可选的 BM25 索引）。

        实现参考 FlexRAG ``MultiVectorRetriever._load_index()``：
        使用 ``OpenAILikeEmbedding`` 作为 Embedding 模型（与构建阶段保持一致），
        通过 ``FaissVectorStore + StorageContext`` 恢复 ``VectorStoreIndex``。
        """
        try:
            import faiss as _faiss
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError:
            raise ImportError(
                "FlexRAGRetriever 需要安装：\n"
                "pip install llama-index llama-index-vector-stores-faiss faiss-cpu httpx"
            )

        faiss_path = os.path.join(self._persist_dir, _FAISS_FILE)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"未找到 FAISS 索引文件：{faiss_path}\n"
                "请先执行 FlexRAG scripts/build_knowledge_base.py 构建知识库，"
                "或将 KNOWLEDGE_PERSIST_DIR 设置为已有知识库的正确目录（含 faiss_index.bin）。"
            )

        # 使用与 temp/embedding.py 相同的 openai SDK 模式初始化 OpenAILikeEmbedding
        self._embed_model = OpenAILikeEmbedding(
            base_url=self._embedding_base_url,
            model_name=self._embedding_model,
            api_key=self._embedding_api_key,
            access_key=self._embedding_access_key,
        )
        Settings.embed_model = self._embed_model

        faiss_index = _faiss.read_index(faiss_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self._persist_dir,
        )
        self._index = load_index_from_storage(storage_context)

        # 可选：加载 BM25 索引
        bm25_path = os.path.join(self._persist_dir, "bm25_index")
        if self._bm25_weight > 0 and os.path.isdir(bm25_path):
            self._load_bm25(bm25_path)

    def _load_bm25(self, bm25_path: str) -> None:
        """加载 BM25 索引（可选）；失败时自动回退到纯向量检索。"""
        try:
            from llama_index.retrievers.bm25 import BM25Retriever as _BM25Retriever
            self._bm25_retriever = _BM25Retriever.from_persist_dir(bm25_path)
        except Exception as exc:
            warnings.warn(
                f"无法加载 BM25 索引（{exc}），回退到纯向量检索。",
                stacklevel=2,
            )
            self._bm25_weight = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        执行向量检索（可选 BM25 混合），返回 :class:`~rag_module.base.RetrievalResult`。

        实现参考 FlexRAG ``MultiVectorRetriever.retrieve()``：
        使用 ``index.as_retriever()`` 获取向量检索结果，
        可选与 BM25 结果按权重融合后排序返回。

        Args:
            query:  检索查询词。
            top_k:  返回片段数量，不传则使用构造时的 ``top_k``。

        Returns:
            :class:`~rag_module.base.RetrievalResult`
        """
        k = top_k if top_k is not None else self._top_k
        dense_w = 1.0 - self._bm25_weight

        # --- 向量检索 ---
        dense_retriever = self._index.as_retriever(similarity_top_k=k)
        dense_nodes = await asyncio.to_thread(dense_retriever.retrieve, query)

        # scored: content_text → (node, combined_score)
        scored: dict[str, tuple[Any, float]] = {}
        for node in dense_nodes:
            key = node.get_content()
            scored[key] = (node, (node.score or 0.0) * dense_w)

        # --- BM25 混合（可选）---
        if self._bm25_retriever is not None and self._bm25_weight > 0:
            bm25_nodes = await asyncio.to_thread(
                self._bm25_retriever.retrieve, query
            )
            for node in bm25_nodes:
                key = node.get_content()
                bm25_score = (node.score or 0.0) * self._bm25_weight
                if key in scored:
                    n, ds = scored[key]
                    scored[key] = (n, ds + bm25_score)
                else:
                    scored[key] = (node, bm25_score)

        # --- 排序 & 封装为 Chunk ---
        sorted_items = sorted(scored.values(), key=lambda x: x[1], reverse=True)[:k]

        chunks: list[Chunk] = []
        for idx, (node, score) in enumerate(sorted_items):
            meta = node.metadata or {}
            source = (
                meta.get("file_name")
                or meta.get("title")
                or meta.get("source")
                or "unknown"
            )
            chunks.append(
                Chunk(
                    chunk_id=str(meta.get("chunk_id", idx)),
                    content=node.get_content(),
                    source=source,
                    score=round(score, 4),
                    metadata=meta,
                )
            )

        return RetrievalResult(
            query=query,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )
