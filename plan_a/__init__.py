"""
plan_a — 基于 Parlant 框架的极简版 RAG 方案。

目录结构按 docs/planA.md 的模块关系组织：

    User ←→ Agent[Parlant Agent]   ← 外部入口（examples/05_plan_a_agent.py）
                     │
                GL[Guidelines]      ← plan_a/guidelines/
               ╱               ╲
    SR[Scoped Retriever]   Tool[RAG Tool]   ← plan_a/retrievers/ / plan_a/tools/
               ╲               ╱
              RAG[RAG 服务]        ← plan_a/rag/
                     │
              VDB[(向量数据库)]    ← plan_a/vdb/

其中 core/ 提供跨模块复用的配置与共享类型。
"""
from plan_a.core import DocChunk, PlanAConfig
from plan_a.rag import Embedder, RAGPipeline
from plan_a.vdb import VectorStore

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


def make_scoped_retriever(*args, **kwargs):
    from plan_a.retrievers import make_scoped_retriever as _impl

    return _impl(*args, **kwargs)


def make_rag_tool(*args, **kwargs):
    from plan_a.tools import make_rag_tool as _impl

    return _impl(*args, **kwargs)


def setup_guidelines(*args, **kwargs):
    from plan_a.guidelines import setup_guidelines as _impl

    return _impl(*args, **kwargs)
