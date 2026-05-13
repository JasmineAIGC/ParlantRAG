"""
KeywordRetriever — 关键词覆盖率检索器。

无需任何外部依赖，适合快速原型验证或小规模知识库。
"""
from __future__ import annotations

import re

from rag_module.retrieval.types import Chunk, RetrievalResult

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    """提取大小写和标点无关的关键词 token。"""
    return {token.lower() for token in _TOKEN_RE.findall(text) if token.strip()}


class KeywordRetriever:
    """
    基于查询关键词覆盖率的轻量检索器。

    对每篇文档计算 ``intersection / query_words`` 得分，并把 ``source`` 一并纳入
    匹配文本，避免短文档在 Jaccard 分母较小时异常靠前。

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
        query_words = _tokenize(query)
        scored: list[tuple[dict, float]] = []

        for doc in self._docs:
            doc_words = _tokenize(f"{doc.get('source', '')} {doc['content']}")
            intersection = len(query_words & doc_words)
            score = round(intersection / len(query_words), 4) if query_words else 0.0
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
