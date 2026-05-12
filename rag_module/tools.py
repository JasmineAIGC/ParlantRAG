"""
Parlant 工具层。

将 Retriever 和 Evaluator 包装为 Parlant 工具。
这一层是 RAG 模块与 Parlant 框架之间的唯一接口。
"""
from __future__ import annotations

import json
from typing import Annotated, Any

import parlant.sdk as p

from .base import BaseRetriever, BaseEvaluator


def make_rag_tools(
    retriever: BaseRetriever,
    evaluator: BaseEvaluator,
) -> tuple[Any, Any]:
    """
    根据传入的检索器和评价器，动态生成两个 Parlant 工具。

    Returns:
        (rag_search_tool, evaluate_retrieval_tool)
    """

    @p.tool
    async def rag_search(
        context: p.ToolContext,
        query: Annotated[
            str,
            p.ToolParameterOptions(
                description="从用户问题中提炼的核心检索词"
            ),
        ],
        top_k: Annotated[
            int,
            p.ToolParameterOptions(
                description="返回片段数量上限，默认 3"
            ),
        ] = 3,
    ) -> p.ToolResult:
        """从知识库检索与查询最相关的文档片段"""
        result = await retriever.retrieve(query=query, top_k=top_k)

        return p.ToolResult(
            data={
                "query": result.query,
                "retrieval_status": result.status,
                "chunk_count": len(result.chunks),
                "retrieved_chunks": [
                    {
                        "chunk_id": c.chunk_id,
                        "content": c.content,
                        "source": c.source,
                        "score": c.score,
                    }
                    for c in result.chunks
                ],
            }
        )

    @p.tool
    async def evaluate_retrieval(
        context: p.ToolContext,
        user_question: Annotated[
            str,
            p.ToolParameterOptions(description="用户的原始问题"),
        ],
        retrieved_chunks_json: Annotated[
            str,
            p.ToolParameterOptions(
                description=(
                    "rag_search 返回的 retrieved_chunks 字段，"
                    "序列化为 JSON 字符串"
                )
            ),
        ],
    ) -> p.ToolResult:
        """评价 RAG 检索结果的质量（相关性、覆盖度、置信度）"""
        from .base import Chunk, RetrievalResult

        try:
            raw_chunks = json.loads(retrieved_chunks_json)
        except (json.JSONDecodeError, TypeError):
            raw_chunks = []

        chunks = [
            Chunk(
                chunk_id=c.get("chunk_id", ""),
                content=c.get("content", ""),
                source=c.get("source", ""),
                score=c.get("score", 0.0),
            )
            for c in raw_chunks
        ]
        retrieval = RetrievalResult(
            query=user_question,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )

        eval_result = await evaluator.evaluate(
            question=user_question,
            retrieval=retrieval,
        )

        return p.ToolResult(
            data={
                "verdict": eval_result.verdict,
                "scores": {
                    "relevance": eval_result.relevance,
                    "coverage": eval_result.coverage,
                    "confidence": eval_result.confidence,
                },
                "conclusion": eval_result.conclusion,
                "action_hint": eval_result.action_hint,
                "details": eval_result.details,
            }
        )

    return rag_search, evaluate_retrieval
