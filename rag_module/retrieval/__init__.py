"""
检索子包。

提供数据类型和开箱即用的检索器实现：

数据类：
  - Chunk             — 检索到的文档片段
  - RetrievalResult   — 检索结果

检索器：
  - KeywordRetriever  — 关键词匹配，无需额外依赖
  - DenseRetriever    — sentence-transformers + FAISS 本地向量检索
  - FlexRAGRetriever  — 加载 FlexRAG 构建的 FAISS 知识库

Embedding：
  - OpenAILikeEmbedding         — OpenAI 兼容 Embedding 客户端
  - SentenceTransformerEmbedding — 本地 sentence-transformers 封装
"""
from rag_module.retrieval.types import Chunk, RetrievalResult
from rag_module.retrieval.embedding import OpenAILikeEmbedding, SentenceTransformerEmbedding
from rag_module.retrieval.keyword import KeywordRetriever
from rag_module.retrieval.dense import DenseRetriever
from rag_module.retrieval.flexrag import FlexRAGRetriever

__all__ = [
    "Chunk",
    "RetrievalResult",
    "OpenAILikeEmbedding",
    "SentenceTransformerEmbedding",
    "KeywordRetriever",
    "DenseRetriever",
    "FlexRAGRetriever",
]
