"""
DenseRetriever — 本地向量检索器（sentence-transformers + FAISS）。

适合生产环境中不依赖外部 Embedding 服务的场景。

依赖::

    pip install sentence-transformers faiss-cpu
"""
from __future__ import annotations

import asyncio
from typing import Any

from rag_module.base import BaseRetriever, Chunk, RetrievalResult


class DenseRetriever(BaseRetriever):
    """
    基于 sentence-transformers 编码、FAISS 内积索引的向量检索器。

    调用 :meth:`build_index` 一次后，后续的 :meth:`retrieve` 均为纯内存操作。

    Usage::

        retriever = DenseRetriever(
            documents=docs,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )
        await retriever.build_index()
        result = await retriever.retrieve("查询词", top_k=5)
    """

    def __init__(
        self,
        documents: list[dict],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        """
        Args:
            documents:   文档列表，每条必须包含 ``id`` / ``content`` / ``source`` 字段。
            model_name:  sentence-transformers 模型名称（或本地路径）。
        """
        self._docs = documents
        self._model_name = model_name
        self._index: Any = None
        self._model: Any = None

    async def build_index(self) -> None:
        """构建向量索引。程序启动时调用一次即可。"""
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

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Args:
            query:  检索查询词
            top_k:  返回片段数量上限

        Raises:
            RuntimeError: 若尚未调用 :meth:`build_index`。
        """
        if self._index is None or self._model is None:
            raise RuntimeError(
                "DenseRetriever 尚未构建索引，请先调用 await retriever.build_index()"
            )

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
