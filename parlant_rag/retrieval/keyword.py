"""
KeywordRetriever — 关键词（Jaccard）检索器。

无需任何外部依赖，适合快速原型验证或小规模知识库。
"""
from __future__ import annotations

from parlant_rag.retrieval.types import Chunk, RetrievalResult


class KeywordRetriever:
    """
    基于 Jaccard 相似度的关键词检索器。

    对每篇文档计算 ``intersection / union`` 词集得分，
    取 top-k 结果返回。

    Usage::

        docs = [
            {"id": "1", "content": "...", "source": "file.md"},
        ]
        retriever = KeywordRetriever(docs)
        result = await retriever.retrieve("查询词", top_k=3)
    """

    def __init__(self, documents: list[dict]) -> None:
        """
        Args:
            documents: 文档列表，每条必须包含 ``id`` / ``content`` / ``source`` 字段。
        """
        self._docs = documents

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        query_words = set(query.lower().split())
        scored: list[tuple[dict, float]] = []

        for doc in self._docs:
            doc_words = set(doc["content"].lower().split())
            union = len(query_words | doc_words)
            intersection = len(query_words & doc_words)
            score = round(intersection / union, 4) if union > 0 else 0.0
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        chunks = [
            Chunk(
                chunk_id=doc["id"],
                content=doc["content"],
                source=doc["source"],
                score=score,
                metadata=doc.get("metadata", {}),
            )
            for doc, score in scored[:top_k]
            if score > 0
        ]

        return RetrievalResult(
            query=query,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )
