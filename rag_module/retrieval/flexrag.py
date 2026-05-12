"""
FlexRAGRetriever — 加载 FlexRAG 构建的 FAISS 向量知识库。

直接读取由 ``FlexRAG scripts/build_knowledge_base.py`` 持久化的索引目录，
无需重新构建。支持纯向量检索和 BM25 混合检索。

依赖::

    pip install llama-index llama-index-vector-stores-faiss faiss-cpu httpx
    # BM25 混合（可选）：
    pip install llama-index-retrievers-bm25 rank_bm25 jieba
"""
from __future__ import annotations

import asyncio
import os
import warnings
from typing import Any, List, Optional

from rag_module.base import BaseRetriever, Chunk, RetrievalResult

_FAISS_FILE = "faiss.index"


class FlexRAGRetriever(BaseRetriever):
    """
    读取 FlexRAG ``MultiVectorRetriever`` 持久化目录，执行向量（+可选 BM25）检索。

    构造时立即加载索引（同步），检索时异步执行。

    Usage::

        retriever = FlexRAGRetriever(
            persist_dir="./knowledge_base",
            embedding_base_url="http://127.0.0.1:8018/v1",
            embedding_api_key="sk-xxxx",
            embedding_model="Qwen3-Embedding-0.6B",
            top_k=5,
            bm25_weight=0.3,   # 0.0 = 纯向量; >0 = 混合（需 bm25_index/ 子目录）
        )
        result = await retriever.retrieve("查询词")
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_base_url: str = "http://127.0.0.1:8018/v1",
        embedding_api_key: str = "sk-xxxx",
        embedding_model: str = "Qwen3-Embedding-0.6B",
        top_k: int = 5,
        bm25_weight: float = 0.0,
    ) -> None:
        """
        Args:
            persist_dir:          FlexRAG 知识库根目录（包含 ``faiss.index`` 等文件）。
            embedding_base_url:   Embedding 服务的 base URL（vLLM / OpenAI 兼容）。
            embedding_api_key:    Embedding 服务的 API Key。
            embedding_model:      Embedding 模型名称。
            top_k:                默认返回的片段数量。
            bm25_weight:          BM25 混合权重（0~1）。0.0 表示纯向量检索。
                                  当 ``persist_dir/bm25_index/`` 存在且此值 > 0 时
                                  自动启用混合检索。
        """
        self._persist_dir = persist_dir
        self._embedding_base_url = embedding_base_url
        self._embedding_api_key = embedding_api_key
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._bm25_weight = bm25_weight

        self._index: Any = None
        self._embed_model: Any = None
        self._bm25_retriever: Any = None

        self._load_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_embed_model(self) -> Any:
        """构造一个最小化的 OpenAI 兼容 Embedding 客户端（不依赖 FlexRAG 包）。"""
        try:
            import httpx
            from llama_index.core.embeddings import BaseEmbedding
            from pydantic import PrivateAttr
        except ImportError:
            raise ImportError(
                "FlexRAGRetriever 需要安装：pip install llama-index httpx"
            )

        raw = self._embedding_base_url.rstrip("/")
        if raw.endswith("/v1/embeddings"):
            endpoint = raw
        elif raw.endswith("/v1"):
            endpoint = raw + "/embeddings"
        else:
            endpoint = raw + "/v1/embeddings"

        _api_key = self._embedding_api_key
        _model_name = self._embedding_model

        class _EmbeddingClient(BaseEmbedding):
            base_url: str
            model_name: str
            api_key: Optional[str] = None
            _client: Any = PrivateAttr()
            _aclient: Any = PrivateAttr()
            _endpoint: str = PrivateAttr()

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                self._endpoint = endpoint          # captured from outer scope
                self._client = httpx.Client(timeout=60.0)
                self._aclient = httpx.AsyncClient(timeout=60.0)

            # -- sync --------------------------------------------------

            def _get_query_embedding(self, query: str) -> List[float]:
                return self._get_text_embeddings([query])[0]

            def _get_text_embedding(self, text: str) -> List[float]:
                return self._get_text_embeddings([text])[0]

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                payload = {
                    "model": self.model_name,
                    "input": texts,
                    "encoding_format": "float",
                }
                headers: dict[str, str] = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                resp = self._client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]

            # -- async -------------------------------------------------

            async def _aget_query_embedding(self, query: str) -> List[float]:
                return (await self._aget_text_embeddings([query]))[0]

            async def _aget_text_embedding(self, text: str) -> List[float]:
                return (await self._aget_text_embeddings([text]))[0]

            async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                payload = {
                    "model": self.model_name,
                    "input": texts,
                    "encoding_format": "float",
                }
                headers: dict[str, str] = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                resp = await self._aclient.post(
                    self._endpoint, json=payload, headers=headers
                )
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]

        return _EmbeddingClient(
            base_url=self._embedding_base_url,
            model_name=_model_name,
            api_key=_api_key,
        )

    def _load_index(self) -> None:
        """加载 FAISS 持久化索引（以及可选的 BM25 索引）。"""
        try:
            import faiss as _faiss
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError:
            raise ImportError(
                "FlexRAGRetriever 需要安装：\n"
                "pip install llama-index llama-index-vector-stores-faiss faiss-cpu"
            )

        faiss_path = os.path.join(self._persist_dir, _FAISS_FILE)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"未找到 FAISS 索引文件：{faiss_path}\n"
                "请先执行 FlexRAG scripts/build_knowledge_base.py 构建知识库。"
            )

        self._embed_model = self._build_embed_model()
        Settings.embed_model = self._embed_model

        faiss_index = _faiss.read_index(faiss_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self._persist_dir,
        )
        self._index = load_index_from_storage(storage_context)

        # Optionally load BM25
        bm25_path = os.path.join(self._persist_dir, "bm25_index")
        if self._bm25_weight > 0 and os.path.isdir(bm25_path):
            self._load_bm25(bm25_path)

    def _load_bm25(self, bm25_path: str) -> None:
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

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """
        向量检索（可选 BM25 混合）。

        Args:
            query:  检索查询词
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

        # --- BM25 混合 ---
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

        # --- 排序 & 封装 ---
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
