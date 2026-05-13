"""
检索子包。

提供数据类型和开箱即用的检索器实现：

数据类：
  - Chunk             — 检索到的文档片段
  - RetrievalResult   — 检索结果

检索器：
  - KeywordRetriever  — 关键词匹配，无需额外依赖
  - DenseRetriever    — sentence-transformers + FAISS 本地向量检索（无 FlexRAG 依赖）
  - ParlantRetriever  — 基于 Parlant SDK 的检索器封装（提供 as_tool() 接口）

Embedding（直接从 model_client 暴露，方便统一导入）：
  - OpenAILikeEmbedding          — OpenAI 兼容 Embedding 客户端
  - SentenceTransformerEmbedding — 本地 sentence-transformers 封装
"""
from rag_module.retrieval.types import Chunk, RetrievalResult
from model_client.api.embedding import OpenAILikeEmbedding
from model_client.local.embedding import SentenceTransformerEmbedding
from rag_module.retrieval.keyword import KeywordRetriever
from rag_module.retrieval.dense import DenseRetriever
from rag_module.retrieval.parlant import ParlantRetriever

__all__ = [
    "Chunk",
    "RetrievalResult",
    "OpenAILikeEmbedding",
    "SentenceTransformerEmbedding",
    "KeywordRetriever",
    "DenseRetriever",
    "ParlantRetriever",
]
