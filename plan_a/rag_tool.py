"""
rag_tool.py — RAG Tool（手动检索工具）组件。

对应架构图中的 ``Tool[RAG Tool]`` 模块。

工作流程::

    Guideline 触发
        │
        ▼ Agent 判断需要检索，主动调用
    search_knowledge(context, query)
        │
        ▼ RAGPipeline.retrieve(query)
    ToolResult（作为工具结果注入 Agent 上下文）

这是**辅助**检索方式：Agent 主动调用，控制更精准，
适合用户明确要求检索或多步推理场景。

使用方式
--------
::

    from plan_a.rag_pipeline import RAGPipeline
    from plan_a.rag_tool import make_rag_tool

    pipeline = RAGPipeline()
    search_knowledge = make_rag_tool(pipeline)

    await agent.create_guideline(
        condition="用户明确要求查询知识库",
        action="调用知识库检索工具后回答",
        tools=[search_knowledge],
    )
"""
from __future__ import annotations

import parlant.sdk as p

from plan_a.rag_pipeline import RAGPipeline
from utils.logging import get_logger


logger = get_logger("plan_a.rag_tool")


def make_rag_tool(pipeline: RAGPipeline):
    """
    工厂函数：创建绑定到 ``pipeline`` 的 Parlant RAG Tool。

    返回值是 ``@p.tool`` 装饰的 ``ToolEntry``，
    可直接传入 ``agent.create_guideline(tools=[...])``。

    Args:
        pipeline: :class:`~plan_a.rag_pipeline.RAGPipeline` 实例（懒初始化安全）。

    Returns:
        ``parlant.sdk.ToolEntry`` — 已注册的 Parlant Tool 对象。
    """

    @p.tool
    async def search_knowledge(context: p.ToolContext, query: str) -> p.ToolResult:
        """
        在知识库中检索与 query 相关的文档片段。

        Args:
            query: 要检索的问题或关键词。

        Returns:
            检索到的知识库内容，按相关性排序。
        """
        logger.info(f"[RagTool] search_knowledge 调用  query={query!r}")

        chunks = await pipeline.retrieve(query)

        if not chunks:
            logger.info("[RagTool] 未命中相关内容")
            return p.ToolResult(
                data={
                    "found": False,
                    "message": "知识库中未找到与该问题相关的内容。",
                    "results": [],
                }
            )

        results = [c.to_dict() for c in chunks]
        formatted = "\n\n".join(
            f"[{i + 1}] 来源: {c.source}\n相关度: {c.score:.3f}\n{c.content}"
            for i, c in enumerate(chunks)
        )

        logger.info(f"[RagTool] 返回 {len(chunks)} 个片段")
        return p.ToolResult(
            data={
                "found": True,
                "chunk_count": len(chunks),
                "formatted": formatted,
                "results": results,
            }
        )

    return search_knowledge
