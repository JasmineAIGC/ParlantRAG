"""
plan_a — 基于 Parlant 框架的「极简版」RAG 实现方案。

完整对应 docs/planA.md 中的极简架构，每个架构组件均有独立文件：

    User ←→ Agent[Parlant Agent]   ← 外部入口（examples/05_plan_a_agent.py），不在本模块内
                     │
                GL[Guidelines]          ← guidelines.py
               ╱               ╲
    SR[Scoped Retriever]   Tool[RAG Tool]   ← scoped_retriever.py / rag_tool.py
               ╲               ╱
              RAG[RAG 服务]             ← rag_pipeline.py
                     │
              VDB[(向量数据库)]         ← vector_store.py

模块文件一览
-----------
- ``config.py``           — 独立配置，从环境变量读取参数
- ``types.py``            — 共享数据类型（DocChunk）
- ``embedding.py``        — 文本向量化（Embedder，基于 model_client）
- ``vector_store.py``     — 向量数据库层（VectorStore / FAISS）
- ``rag_pipeline.py``     — RAG 检索管道（核心服务，SR 和 Tool 共用）
- ``scoped_retriever.py`` — Scoped Retriever（自动检索，主力方式）
- ``rag_tool.py``         — RAG Tool（手动检索，辅助方式）
- ``guidelines.py``       — Guidelines 配置层（GL）

快速上手
--------
::

    # 外部 Agent 入口（see examples/05_plan_a_agent.py）
    import parlant.sdk as p
    from plan_a import PlanAConfig, RAGPipeline, setup_guidelines

    async def main():
        pipeline = RAGPipeline(PlanAConfig.from_env())
        await pipeline.initialize()

        async with p.Server() as server:
            agent = await server.create_agent(name="...", description="...")
            await setup_guidelines(agent, pipeline)
"""
from plan_a.config import PlanAConfig
from plan_a.types import DocChunk
from plan_a.embedding import Embedder
from plan_a.vector_store import VectorStore
from plan_a.rag_pipeline import RAGPipeline
from plan_a.scoped_retriever import make_scoped_retriever
from plan_a.rag_tool import make_rag_tool
from plan_a.guidelines import setup_guidelines

__all__ = [
    # 配置 & 类型
    "PlanAConfig",
    "DocChunk",
    # 基础组件
    "Embedder",
    "VectorStore",
    # RAG 管道（核心）
    "RAGPipeline",
    # Parlant 集成
    "make_scoped_retriever",
    "make_rag_tool",
    "setup_guidelines",
]
