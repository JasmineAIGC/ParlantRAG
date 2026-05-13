"""
parlant_rag — ParlantRAG 核心模块。

模块结构：
  parlant_rag/retrieval/   — 检索（Chunk, RetrievalResult, FlexRAGRetriever 等）
  parlant_rag/evaluation/  — 评估（EvaluationResult, LLMEvaluator）
  parlant_rag/tools/       — Parlant 集成工具（make_rag_tools, RAGModule）

模型调用（实现位于 model_client/）：
  model_client/api/        — 通过 API 调用 LLM 和 Embedding 服务
  model_client/local/      — 本地加载模型（sentence-transformers）

所有 API 配置均从项目根目录 .env 文件自动加载。

Parlant 集成（可选，需安装 parlant）：
  parlant_rag/tools/module.py       — RAGModule
  parlant_rag/tools/parlant_tools.py — make_rag_tools
"""
from model_client.api.llm import LLMClient, LLMConfig
from parlant_rag.retrieval import Chunk, RetrievalResult, FlexRAGRetriever
from parlant_rag.evaluation import EvaluationResult, LLMEvaluator

# Parlant 集成（可选）：仅当 parlant 已安装时才可用。
try:
    from parlant_rag.tools.module import RAGModule
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
