"""
rag_module — RAG 检索与评估模块。

对外暴露的公共接口。

子包划分：
  rag_module/retrieval/   — 检索相关实现
  rag_module/evaluation/  — 评估相关实现
  rag_module/module.py    — Parlant Agent 集成（可选，需安装 parlant）
"""
from rag_module.base import (
    BaseRetriever,
    BaseEvaluator,
    Chunk,
    RetrievalResult,
    RetrievalSignals,
    EvaluationResult,
)
from rag_module.llm_client import LLMClient, LLMConfig
from rag_module.retrieval import FlexRAGRetriever
from rag_module.evaluation import LLMEvaluator

# Parlant 集成（可选）：仅当 parlant 已安装时才可用。
# 纯检索 / 评估场景无需 parlant，直接使用 FlexRAGRetriever + LLMEvaluator 即可。
try:
    from rag_module.module import RAGModule
except ImportError:
    RAGModule = None  # type: ignore[assignment,misc]

__all__ = [
    # 抽象接口
    "BaseRetriever",
    "BaseEvaluator",
    # 数据类
    "Chunk",
    "RetrievalResult",
    "RetrievalSignals",
    "EvaluationResult",
    # LLM 客户端（参考 temp/llm/llm_client.py 的 AdkLLMClient 模式）
    "LLMClient",
    "LLMConfig",
    # 检索器（知识库调用参考 FlexRAG MultiVectorRetriever）
    "FlexRAGRetriever",
    # 评估器
    "LLMEvaluator",
    # Parlant 集成（可选）
    "RAGModule",
]
