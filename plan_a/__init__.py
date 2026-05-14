"""
plan_a — 基于 Parlant 框架的「极简版」RAG Agent 实现。

对应 docs/planA.md 中描述的极简架构：

    用户 <→ Parlant Agent → Guidelines
                              ├─ Scoped Retriever（自动触发，主力方式）
                              └─ RAG Tool（手动调用，辅助方式）
                                        ↕
                                    RAG 服务
                                        ↕
                                    向量数据库

核心组件
-------
- ``RagService``         : 封装检索管道（Embedding → FAISS → 返回文档片段）
- ``make_rag_tool``      : 生成 ``@p.tool`` 装饰的知识库检索函数
- ``build_plan_a_agent`` : 一键创建带 Guideline + Scoped Retriever + RAG Tool 的 Parlant Agent

快速上手
-------
::

    import parlant.sdk as p
    from plan_a import RagService, build_plan_a_agent

    async def main():
        rag = RagService()
        await rag.initialize()

        async with p.Server() as server:
            agent = await build_plan_a_agent(server, rag)
            # Agent 已挂载 Scoped Retriever 和 RAG Tool，可接受用户会话
"""
from plan_a.rag_service import RagService
from plan_a.tools import make_rag_tool
from plan_a.agent_builder import build_plan_a_agent

__all__ = [
    "RagService",
    "make_rag_tool",
    "build_plan_a_agent",
]
