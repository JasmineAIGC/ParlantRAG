"""
rag_module — RAG 检索与评估模块。

结构：
  rag_module/retrieval/   — 检索（Chunk, RetrievalResult, FlexRAGRetriever 等）
  rag_module/evaluator.py — 评估（EvaluationResult, LLMEvaluator）

模型调用（实现位于 model_client/）：
  model_client/api/       — 通过 API 调用 LLM 和 Embedding 服务
  model_client/local/     — 本地加载模型（sentence-transformers）

Parlant 集成（可选，需安装 parlant）：
  rag_module/module.py    — RAGModule
  rag_module/tools.py     — make_rag_tools
"""
from model_client.api.llm import LLMClient, LLMConfig
from rag_module.retrieval import Chunk, RetrievalResult, FlexRAGRetriever
from rag_module.evaluator import EvaluationResult, LLMEvaluator

# Parlant 集成（可选）：仅当 parlant 已安装时才可用。
try:
    from rag_module.module import RAGModule
except ImportError:
    RAGModule = None  # type: ignore[assignment,misc]

__all__ = [
    # 数据类
    "Chunk",
    "RetrievalResult",
    "EvaluationResult",
    # LLM 客户端
    "LLMClient",
    "LLMConfig",
    # 检索器
    "FlexRAGRetriever",
    # 评估器
    "LLMEvaluator",
    # Parlant 集成（可选）
    "RAGModule",
]
