"""
Agent 构建器 — 一键创建完整 Plan A 架构的 Parlant Agent。

对应 docs/planA.md 极简架构，本模块负责将各组件连接在一起：

    Parlant Agent
        │
        ├─ Guideline 1: 自动检索（Scoped Retriever 挂载）
        │       condition : 用户询问知识库相关内容
        │       action    : 检索知识库后回答
        │       retriever : RagService.as_retriever_fn()
        │
        └─ Guideline 2: 手动检索（RAG Tool 挂载）
                condition : 用户明确要求检索或查询
                action    : 调用知识库检索工具后回答
                tools     : [make_rag_tool(rag_service)]

两种知识获取路径
----------------
1. **Scoped Retriever（主力）**：Guideline 匹配时 Parlant 引擎
   自动调用 retriever 函数，将检索结果注入 Agent 上下文，无需
   Agent 显式调用工具——延迟低、对话感更自然。

2. **RAG Tool（辅助）**：Agent 在判断需要时主动调用 Tool，
   检索结果以工具返回值形式出现在上下文中——控制力更强，
   适合多步推理或用户明确要求检索的场景。

使用方式
--------
::

    import parlant.sdk as p
    from plan_a import RagService, build_plan_a_agent

    async def main():
        rag = RagService()
        await rag.initialize()

        async with p.Server() as server:
            agent = await build_plan_a_agent(server, rag)
            # 服务持续运行，接受用户会话...
"""
from __future__ import annotations

from typing import Optional

import parlant.sdk as p

from plan_a.rag_service import RagService
from plan_a.tools import make_rag_tool
from utils.logging import get_logger


logger = get_logger("plan_a.agent_builder")


# ---------------------------------------------------------------------------
# 默认配置常量
# ---------------------------------------------------------------------------

_DEFAULT_AGENT_NAME = "KnowledgeAssistant"
_DEFAULT_AGENT_DESCRIPTION = (
    "专业的知识库问答助手。"
    "当用户提出问题时，我会自动检索相关知识库内容并给出准确、简洁的回答。"
    "如果知识库中没有相关内容，我会如实告知。"
)

# Scoped Retriever Guideline（自动检索）
_SCOPED_RETRIEVER_CONDITION = (
    "用户提问涉及知识库内容，例如政策、产品信息、会员权益、"
    "退货规则、服务说明或其他业务知识"
)
_SCOPED_RETRIEVER_ACTION = (
    "自动检索知识库，将检索到的相关内容作为上下文，基于该内容给出准确回答；"
    "若检索内容不足，说明知识库暂无相关信息"
)

# RAG Tool Guideline（手动检索）
_RAG_TOOL_CONDITION = (
    "用户明确要求检索、搜索或查询知识库内容，"
    "例如：'帮我查…'、'搜索一下…'、'查找关于…的信息'"
)
_RAG_TOOL_ACTION = (
    "调用知识库检索工具精准搜索相关文档，"
    "并基于检索结果给出详细、有来源依据的回答"
)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

async def build_plan_a_agent(
    server,
    rag_service: RagService,
    agent_name: str = _DEFAULT_AGENT_NAME,
    agent_description: str = _DEFAULT_AGENT_DESCRIPTION,
    scoped_retriever_condition: Optional[str] = None,
    scoped_retriever_action: Optional[str] = None,
    rag_tool_condition: Optional[str] = None,
    rag_tool_action: Optional[str] = None,
):
    """
    创建完整 Plan A 架构的 Parlant Agent。

    该函数完成以下步骤：

    1. 使用 ``server.create_agent()`` 创建 Parlant Agent。
    2. 创建「自动检索」Guideline，并通过
       ``guideline.attach_retriever()`` 挂载 Scoped Retriever。
    3. 创建「手动检索」Guideline，并将 RAG Tool 绑定为可用工具。

    Args:
        server:
            Parlant ``Server`` 实例（``async with p.Server() as server`` 中的对象）。
        rag_service:
            已初始化的 :class:`~plan_a.rag_service.RagService` 实例。
        agent_name:
            Agent 显示名称，默认 ``"KnowledgeAssistant"``。
        agent_description:
            Agent 行为描述，Parlant 引擎据此生成响应风格。
        scoped_retriever_condition:
            覆盖自动检索 Guideline 的触发条件（留空使用内置默认值）。
        scoped_retriever_action:
            覆盖自动检索 Guideline 的执行动作（留空使用内置默认值）。
        rag_tool_condition:
            覆盖手动检索 Guideline 的触发条件（留空使用内置默认值）。
        rag_tool_action:
            覆盖手动检索 Guideline 的执行动作（留空使用内置默认值）。

    Returns:
        ``parlant.sdk.Agent`` — 配置完毕、可接受会话的 Parlant Agent。

    Examples
    --------
    ::

        import parlant.sdk as p
        from plan_a import RagService, build_plan_a_agent

        async def main():
            rag = RagService()
            await rag.initialize()

            async with p.Server() as server:
                agent = await build_plan_a_agent(server, rag)
                # server 持续运行，开始接受用户会话
    """
    logger.info(f"[AgentBuilder] 创建 Agent: {agent_name!r}")

    # 1. 创建 Parlant Agent
    agent = await server.create_agent(
        name=agent_name,
        description=agent_description,
    )
    logger.info(f"[AgentBuilder] Agent 创建完成  id={agent.id}")

    # ------------------------------------------------------------------
    # 2. Guideline 1：Scoped Retriever（自动检索，主力方式）
    # ------------------------------------------------------------------
    sr_condition = scoped_retriever_condition or _SCOPED_RETRIEVER_CONDITION
    sr_action = scoped_retriever_action or _SCOPED_RETRIEVER_ACTION

    logger.info("[AgentBuilder] 创建 Scoped Retriever Guideline")
    scoped_guideline = await agent.create_guideline(
        condition=sr_condition,
        action=sr_action,
    )

    # 将 RagService 的检索函数挂载到该 Guideline
    # 每次 Guideline 被匹配时，Parlant 引擎自动调用 retriever_fn
    # 并将结果注入 Agent 的推理上下文
    retriever_fn = rag_service.as_retriever_fn()
    await scoped_guideline.attach_retriever(retriever_fn)

    logger.info(
        f"[AgentBuilder] Scoped Retriever 挂载完成  guideline_id={scoped_guideline.id}"
    )

    # ------------------------------------------------------------------
    # 3. Guideline 2：RAG Tool（手动检索，辅助方式）
    # ------------------------------------------------------------------
    tool_condition = rag_tool_condition or _RAG_TOOL_CONDITION
    tool_action = rag_tool_action or _RAG_TOOL_ACTION

    logger.info("[AgentBuilder] 创建 RAG Tool Guideline")
    search_knowledge = make_rag_tool(rag_service)
    rag_tool_guideline = await agent.create_guideline(
        condition=tool_condition,
        action=tool_action,
        tools=[search_knowledge],
    )

    logger.info(
        f"[AgentBuilder] RAG Tool 挂载完成  guideline_id={rag_tool_guideline.id}"
    )

    logger.info(
        f"[AgentBuilder] Plan A Agent 构建完成\n"
        f"  Agent       : {agent_name} (id={agent.id})\n"
        f"  Guidelines  : Scoped Retriever + RAG Tool\n"
        f"  RAG Service : {rag_service.__class__.__name__}"
    )

    return agent
