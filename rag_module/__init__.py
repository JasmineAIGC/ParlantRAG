"""
对外暴露的公共接口。

宿主系统只需要从这里导入，不需要了解内部模块结构。
"""
from .module import RAGModule
from .base import BaseRetriever, BaseEvaluator, Chunk, RetrievalResult, EvaluationResult
from .retriever import KeywordRetriever, DenseRetriever, FlexRAGRetriever
from .evaluator import RuleBasedEvaluator, LLMEvaluator

__all__ = [
    # 主入口
    "RAGModule",
    # 抽象接口（用于自定义实现）
    "BaseRetriever",
    "BaseEvaluator",
    # 数据类
    "Chunk",
    "RetrievalResult",
    "EvaluationResult",
    # 内置检索器
    "KeywordRetriever",
    "DenseRetriever",
    "FlexRAGRetriever",
    # 内置评价器
    "RuleBasedEvaluator",
    "LLMEvaluator",
]
