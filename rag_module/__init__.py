"""
rag_module — Parlant RAG 检索与评估模块。

对外暴露的公共接口。宿主系统只需从这里导入，不需要了解内部子包结构。

子包划分：
  rag_module/retrieval/   — 检索相关实现
  rag_module/evaluation/  — 评估相关实现
"""
from rag_module.module import RAGModule
from rag_module.base import (
    BaseRetriever,
    BaseEvaluator,
    Chunk,
    RetrievalResult,
    RetrievalSignals,
    EvaluationResult,
)
from rag_module.retrieval import KeywordRetriever, DenseRetriever, FlexRAGRetriever
from rag_module.evaluation import LLMEvaluator

__all__ = [
    # 主入口
    "RAGModule",
    # 抽象接口
    "BaseRetriever",
    "BaseEvaluator",
    # 数据类
    "Chunk",
    "RetrievalResult",
    "RetrievalSignals",
    "EvaluationResult",
    # 检索器
    "KeywordRetriever",
    "DenseRetriever",
    "FlexRAGRetriever",
    # 评估器
    "LLMEvaluator",
]
