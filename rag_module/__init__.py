"""
rag_module — Parlant SDK 原生 RAG 模块。

这是替代旧 ``parlant_rag`` / FlexRAG 集成的新模块路径。检索、评估和
Parlant 工具集成均不再依赖 FlexRAG；LLM 与 Embedding 调用仍复用
``model_client`` 中既有的 API / 本地模型客户端。
"""
from model_client.api.llm import LLMClient, LLMConfig
from rag_module.evaluation import EvaluationResult, LLMEvaluator, RetrievalSignals
from rag_module.retrieval import (
    Chunk,
    DenseRetriever,
    KeywordRetriever,
    OpenAILikeEmbedding,
    ParlantRetriever,
    RetrievalResult,
    SentenceTransformerEmbedding,
)

try:
    from rag_module.tools import RAGModule, make_rag_tools
except ImportError:
    RAGModule = None  # type: ignore[assignment,misc]
    make_rag_tools = None  # type: ignore[assignment,misc]

__all__ = [
    "Chunk",
    "RetrievalResult",
    "EvaluationResult",
    "RetrievalSignals",
    "LLMClient",
    "LLMConfig",
    "OpenAILikeEmbedding",
    "SentenceTransformerEmbedding",
    "KeywordRetriever",
    "DenseRetriever",
    "ParlantRetriever",
    "LLMEvaluator",
    "RAGModule",
    "make_rag_tools",
]
