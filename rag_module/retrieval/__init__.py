"""
检索子包。

提供三个开箱即用的检索器实现，均继承自 BaseRetriever：

  - KeywordRetriever  — 关键词匹配，无需额外依赖
  - DenseRetriever    — sentence-transformers + FAISS 本地向量检索
  - FlexRAGRetriever  — 加载 FlexRAG build_knowledge_base.py 构建的 FAISS 知识库

Embedding：

  - OpenAILikeEmbedding         — OpenAI 兼容 Embedding 客户端（LlamaIndex BaseEmbedding 子类）
  - SentenceTransformerEmbedding — 本地 sentence-transformers 封装（LlamaIndex BaseEmbedding 子类）
"""
from rag_module.retrieval.embedding import OpenAILikeEmbedding, SentenceTransformerEmbedding
from rag_module.retrieval.keyword import KeywordRetriever
from rag_module.retrieval.dense import DenseRetriever
from rag_module.retrieval.flexrag import FlexRAGRetriever

__all__ = [
    "OpenAILikeEmbedding",
    "SentenceTransformerEmbedding",
    "KeywordRetriever",
    "DenseRetriever",
    "FlexRAGRetriever",
]
