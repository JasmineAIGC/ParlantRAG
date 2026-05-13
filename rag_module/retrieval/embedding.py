"""
Embedding 模型封装 — 支持 OpenAI 兼容 API 和本地 sentence-transformers 两种模式。

实际实现位于：
  - :mod:`model_client.api.embedding`   — API 模式（OpenAILikeEmbedding）
  - :mod:`model_client.local.embedding` — 本地模式（SentenceTransformerEmbedding）

此处重新导出以保持向后兼容性。

Usage::

    from rag_module.retrieval.embedding import OpenAILikeEmbedding, SentenceTransformerEmbedding
    # 或直接从独立模块导入：
    from model_client.api import OpenAILikeEmbedding
    from model_client.local import SentenceTransformerEmbedding
"""
from model_client.api.embedding import OpenAILikeEmbedding
from model_client.local.embedding import SentenceTransformerEmbedding

__all__ = ["OpenAILikeEmbedding", "SentenceTransformerEmbedding"]
