"""
内置检索器实现。

提供三个开箱即用的实现：
  - KeywordRetriever：基于关键词匹配，无需额外依赖，适合快速验证
  - DenseRetriever：基于 sentence-transformers + FAISS，适合生产
  - FlexRAGRetriever：加载由 FlexRAG scripts/build_knowledge_base.py
                      构建好的 FAISS 向量知识库，适合与 FlexRAG 配套使用

宿主系统也可以完全忽略这个文件，自己实现 BaseRetriever。
"""
from __future__ import annotations

import asyncio
import os
from typing import Any, List, Optional

from .base import BaseRetriever, Chunk, RetrievalResult


# ---------------------------------------------------------------------------
# KeywordRetriever — 开箱即用，无需额外依赖
# ---------------------------------------------------------------------------

class KeywordRetriever(BaseRetriever):
    """
    关键词检索器（无需额外依赖）。

    适合：快速原型验证、小规模知识库、不想引入向量库的场景。

    Usage::

        docs = [{"id": "1", "content": "...", "source": "file.md"}]
        retriever = KeywordRetriever(docs)
    """

    def __init__(self, documents: list[dict]) -> None:
        """
        Args:
            documents: 文档列表，每条必须包含 id / content / source 字段。
        """
        self._docs = documents

    async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        query_words = set(query.lower().split())
        scored: list[tuple[dict, float]] = []

        for doc in self._docs:
            doc_words = set(doc["content"].lower().split())
            union = len(query_words | doc_words)
            intersection = len(query_words & doc_words)
            score = round(intersection / union, 4) if union > 0 else 0.0
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        chunks = [
            Chunk(
                chunk_id=doc["id"],
                content=doc["content"],
                source=doc["source"],
                score=score,
                metadata=doc.get("metadata", {}),
            )
            for doc, score in top
            if score > 0
        ]

        return RetrievalResult(
            query=query,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )


# ---------------------------------------------------------------------------
# DenseRetriever — sentence-transformers + FAISS
# ---------------------------------------------------------------------------

class DenseRetriever(BaseRetriever):
    """
    向量检索器（需要 sentence-transformers + faiss-cpu）。

    适合：生产环境、大规模知识库、需要语义匹配的场景。

    Usage::

        retriever = DenseRetriever(
            documents=docs,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )
        await retriever.build_index()   # 启动时调用一次
    """

    def __init__(
        self,
        documents: list[dict],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self._docs = documents
        self._model_name = model_name
        self._index: Any = None
        self._model: Any = None

    async def build_index(self) -> None:
        """构建向量索引，启动时调用一次。"""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "DenseRetriever 需要安装额外依赖：\n"
                "pip install sentence-transformers faiss-cpu"
            )

        def _build() -> None:
            self._model = SentenceTransformer(self._model_name)
            texts = [doc["content"] for doc in self._docs]
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(dim)
            self._index.add(embeddings.astype(np.float32))

        await asyncio.to_thread(_build)

    async def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        if self._index is None or self._model is None:
            raise RuntimeError("请先调用 await retriever.build_index()")

        import numpy as np

        query_vec = self._model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        scores, indices = self._index.search(query_vec, top_k)

        chunks = [
            Chunk(
                chunk_id=self._docs[idx]["id"],
                content=self._docs[idx]["content"],
                source=self._docs[idx]["source"],
                score=round(float(scores[0][i]), 4),
                metadata=self._docs[idx].get("metadata", {}),
            )
            for i, idx in enumerate(indices[0])
            if idx >= 0 and scores[0][i] > 0
        ]

        return RetrievalResult(
            query=query,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )


# ---------------------------------------------------------------------------
# FlexRAGRetriever — 加载 FlexRAG build_knowledge_base.py 构建的知识库
# ---------------------------------------------------------------------------

class FlexRAGRetriever(BaseRetriever):
    """
    加载由 ``FlexRAG scripts/build_knowledge_base.py`` 构建的 FAISS 向量知识库。

    与 FlexRAG 的 ``MultiVectorRetriever`` 完全兼容；直接读取其持久化目录即可，
    无需重新构建索引。

    依赖::

        pip install llama-index llama-index-vector-stores-faiss faiss-cpu httpx

    Usage::

        retriever = FlexRAGRetriever(
            persist_dir="./knowledge_base",
            embedding_base_url="http://127.0.0.1:8018/v1",
            embedding_api_key="sk-xxxx",
            embedding_model="Qwen3-Embedding-0.6B",
        )
        # 无需 build_index，索引已由 FlexRAG 构建好
    """

    _FAISS_FILE = "faiss.index"

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
            persist_dir:          FlexRAG 知识库目录（包含 faiss.index 等文件）。
            embedding_base_url:   Embedding 服务的 base URL。
            embedding_api_key:    Embedding 服务的 API Key。
            embedding_model:      Embedding 模型名称。
            top_k:                默认返回的片段数量。
            bm25_weight:          BM25 混合权重（0.0 表示纯向量检索）。
                                  当 persist_dir 内存在 bm25_index/ 且此值 > 0 时启用混合检索。
        """
        self._persist_dir = persist_dir
        self._embedding_base_url = embedding_base_url
        self._embedding_api_key = embedding_api_key
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._bm25_weight = bm25_weight

        self._index: Any = None
        self._embed_model: Any = None
        self._retriever: Any = None
        self._bm25_retriever: Any = None

        self._load()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_embed_model(self) -> Any:
        """Build a minimal OpenAI-compatible embedding client (no FlexRAG dep)."""
        try:
            import httpx
            from llama_index.core.embeddings import BaseEmbedding
            from pydantic import PrivateAttr
        except ImportError:
            raise ImportError(
                "FlexRAGRetriever 需要安装：pip install llama-index httpx"
            )

        raw_base_url = self._embedding_base_url.rstrip("/")
        if raw_base_url.endswith("/v1/embeddings"):
            endpoint = raw_base_url
        elif raw_base_url.endswith("/v1"):
            endpoint = raw_base_url + "/embeddings"
        else:
            endpoint = raw_base_url + "/v1/embeddings"

        api_key = self._embedding_api_key
        model_name = self._embedding_model

        class _InlineEmbedding(BaseEmbedding):
            base_url: str
            model_name: str
            api_key: Optional[str] = None
            _client: Any = PrivateAttr()
            _aclient: Any = PrivateAttr()
            _endpoint: str = PrivateAttr()

            def __init__(self, **kwargs: Any) -> None:
                super().__init__(**kwargs)
                # endpoint is resolved once by the outer factory method
                self._endpoint = endpoint
                self._client = httpx.Client(timeout=60.0)
                self._aclient = httpx.AsyncClient(timeout=60.0)

            def _get_query_embedding(self, query: str) -> List[float]:
                return self._get_text_embeddings([query])[0]

            def _get_text_embedding(self, text: str) -> List[float]:
                return self._get_text_embeddings([text])[0]

            def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                payload = {"model": self.model_name, "input": texts, "encoding_format": "float"}
                headers: dict[str, str] = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                resp = self._client.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]

            async def _aget_query_embedding(self, query: str) -> List[float]:
                res = await self._aget_text_embeddings([query])
                return res[0]

            async def _aget_text_embedding(self, text: str) -> List[float]:
                res = await self._aget_text_embeddings([text])
                return res[0]

            async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                payload = {"model": self.model_name, "input": texts, "encoding_format": "float"}
                headers: dict[str, str] = {}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                resp = await self._aclient.post(self._endpoint, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()["data"]
                data.sort(key=lambda item: item["index"])
                return [item["embedding"] for item in data]

        return _InlineEmbedding(
            base_url=self._embedding_base_url,
            model_name=model_name,
            api_key=api_key,
        )

    def _load(self) -> None:
        """Load the persisted FAISS index and (optionally) the BM25 index."""
        try:
            import faiss as _faiss
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError:
            raise ImportError(
                "FlexRAGRetriever 需要安装：\n"
                "pip install llama-index llama-index-vector-stores-faiss faiss-cpu"
            )

        faiss_path = os.path.join(self._persist_dir, self._FAISS_FILE)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"未找到 FAISS 索引文件：{faiss_path}\n"
                "请先使用 FlexRAG scripts/build_knowledge_base.py 构建知识库。"
            )

        self._embed_model = self._make_embed_model()
        Settings.embed_model = self._embed_model

        faiss_index = _faiss.read_index(faiss_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self._persist_dir,
        )
        self._index = load_index_from_storage(storage_context)

        # Try to load BM25 index if available and weight > 0
        bm25_path = os.path.join(self._persist_dir, "bm25_index")
        if self._bm25_weight > 0 and os.path.isdir(bm25_path):
            self._load_bm25(bm25_path)

    def _load_bm25(self, bm25_path: str) -> None:
        try:
            from llama_index.retrievers.bm25 import BM25Retriever as _BM25Retriever
            self._bm25_retriever = _BM25Retriever.from_persist_dir(bm25_path)
        except Exception as exc:
            # BM25 is optional — log and fall back to dense-only
            import warnings
            warnings.warn(
                f"无法加载 BM25 索引（{exc}），将使用纯向量检索。",
                stacklevel=2,
            )
            self._bm25_weight = 0.0

    # ------------------------------------------------------------------
    # Retrieve
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        k = top_k if top_k is not None else self._top_k

        dense_retriever = self._index.as_retriever(similarity_top_k=k)
        dense_nodes = await asyncio.to_thread(dense_retriever.retrieve, query)

        # Build a score map: text → (node, dense_score)
        scored: dict[str, tuple[Any, float]] = {}
        dense_w = 1.0 - self._bm25_weight

        for node in dense_nodes:
            key = node.get_content()
            scored[key] = (node, (node.score or 0.0) * dense_w)

        # Hybrid: add BM25 scores
        if self._bm25_retriever is not None and self._bm25_weight > 0:
            bm25_nodes = await asyncio.to_thread(self._bm25_retriever.retrieve, query)
            for node in bm25_nodes:
                key = node.get_content()
                bm25_score = (node.score or 0.0) * self._bm25_weight
                if key in scored:
                    n, ds = scored[key]
                    scored[key] = (n, ds + bm25_score)
                else:
                    scored[key] = (node, bm25_score)

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
            chunk_id = meta.get("chunk_id", str(idx))
            chunks.append(
                Chunk(
                    chunk_id=str(chunk_id),
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
