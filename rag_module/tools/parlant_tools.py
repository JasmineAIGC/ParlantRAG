"""
Parlant SDK 工具工厂。

将检索器与评估器封装为两个官方 Parlant ``@p.tool`` 工具：
``rag_search`` 和 ``evaluate_retrieval``。工具返回 ``p.ToolResult``，可直接
挂载到 ``agent.create_guideline(..., tools=[...])`` 或 ``agent.attach_tool(...)``。
"""
import json
from typing import Annotated, Any

import parlant.sdk as p

from rag_module.retrieval.types import Chunk, RetrievalResult


def make_rag_tools(retriever: Any, evaluator: Any) -> tuple[Any, Any]:
    """
    根据传入的检索器和评估器动态生成 Parlant SDK 工具。

    Args:
        retriever: 实现 ``retrieve(query, top_k)`` 异步方法的检索器。
        evaluator: 实现 ``evaluate(question, retrieval)`` 异步方法的评估器。

    Returns:
        ``(rag_search, evaluate_retrieval)`` 两个可注册到 Parlant Agent 的工具。
    """

    @p.tool
    async def rag_search(
        context: p.ToolContext,
        query: Annotated[
            str,
            p.ToolParameterOptions(description="从用户问题中提炼的核心检索词"),
        ],
        top_k: Annotated[
            int,
            p.ToolParameterOptions(description="返回片段数量上限，默认 5"),
        ] = 5,
    ) -> p.ToolResult:
        """从知识库检索与查询最相关的文档片段。"""
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
                        "metadata": c.metadata,
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
                description="rag_search 返回的 retrieved_chunks 字段，序列化为 JSON 字符串"
            ),
        ],
    ) -> p.ToolResult:
        """对 RAG 检索结果进行多信号 LLM 综合评估。"""
        try:
            raw_chunks = json.loads(retrieved_chunks_json)
        except (json.JSONDecodeError, TypeError):
            raw_chunks = []

        chunks = [
            Chunk(
                chunk_id=c.get("chunk_id", ""),
                content=c.get("content", ""),
                source=c.get("source", ""),
                score=float(c.get("score", 0.0)),
                metadata=c.get("metadata", {}) if isinstance(c.get("metadata", {}), dict) else {},
            )
            for c in raw_chunks
            if isinstance(c, dict)
        ]
        retrieval = RetrievalResult(
            query=user_question,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )

        eval_result = await evaluator.evaluate(question=user_question, retrieval=retrieval)

        signals_data: dict[str, Any] = {}
        if eval_result.signals is not None:
            s = eval_result.signals
            signals_data = {
                "avg_score": s.avg_score,
                "max_score": s.max_score,
                "min_score": s.min_score,
                "keyword_coverage": s.keyword_coverage,
                "source_diversity": s.source_diversity,
                "chunk_count": s.chunk_count,
                "missing_keywords": s.missing_keywords,
            }

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
                "signals": signals_data,
            }
        )

    return rag_search, evaluate_retrieval
