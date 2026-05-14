"""
RAG Tool 工厂 — 生成可挂载到 Parlant Guideline 的 ``@p.tool`` 检索函数。

设计思路
--------
Parlant 的 Tool（工具）是**辅助**知识获取方式，由 Agent 主动调用。
与 Scoped Retriever（自动触发）相比，Tool 让 Agent 能在对话流程中
按需精准检索特定问题的知识库内容。

典型场景
--------
- 用户明确说"帮我查…"、"搜索一下…"等关键词时
- Agent 判断需要更精准的检索（而非依赖 Scoped Retriever 的自动检索）
- 多步骤对话中需要针对子问题分别检索

使用方式
--------
::

    from plan_a.tools import make_rag_tool
    from plan_a.rag_service import RagService

    rag = RagService()
    await rag.initialize()

    search_knowledge = make_rag_tool(rag)

    # 将 Tool 挂载到 Guideline
    await agent.create_guideline(
        condition="用户明确要求检索或查询特定知识内容",
        action="调用知识库检索工具搜索相关信息并回答",
        tools=[search_knowledge],
    )
"""
from __future__ import annotations

import parlant.sdk as p

from plan_a.rag_service import RagService
from utils.logging import get_logger


logger = get_logger("plan_a.tools")


def make_rag_tool(rag_service: RagService):
    """
    工厂函数：创建一个绑定到指定 ``RagService`` 的 Parlant RAG Tool。

    返回的函数已用 ``@p.tool`` 装饰，可直接传入
    ``agent.create_guideline(tools=[...])``。

    Args:
        rag_service: 已初始化（或懒初始化）的 :class:`RagService` 实例。

    Returns:
        ``parlant.sdk.ToolEntry`` — 已注册的 Parlant Tool 对象。

    Examples
    --------
    ::

        rag = RagService()
        search_knowledge = make_rag_tool(rag)

        await agent.create_guideline(
            condition="用户要求查询知识库",
            action="调用知识库检索工具后回答",
            tools=[search_knowledge],
        )
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
        logger.info(f"[RagTool] search_knowledge 被调用  query={query!r}")

        chunks = await rag_service.retrieve(query)

        if not chunks:
            logger.info("[RagTool] 未检索到相关内容")
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
