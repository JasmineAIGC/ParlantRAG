"""
embedding.py — 文本向量化组件（Embedder）。

对应架构图中 RAG 服务的 Embedding 步骤。
底层通过 OpenAI-compatible API 调用远程 Embedding 服务，
直接使用 model_client.OpenAILikeEmbedding（共享基础设施），
不依赖 parlant_rag/ 或其他 RAG 实现模块。

配置优先级（同 model_client）：
  EMBEDDING_BASE_URL > QWEN3_BASE_URL > LLM_BASE_URL
"""
from __future__ import annotations

from typing import List, Optional

from model_client import OpenAILikeEmbedding
from utils.logging import get_logger


logger = get_logger("plan_a.embedding")


class Embedder:
    """
    文本向量化器。

    封装 ``model_client.OpenAILikeEmbedding``，提供简洁的异步接口：

    - :meth:`embed_query`  — 向量化单条查询文本
    - :meth:`embed_texts`  — 批量向量化文本列表

    客户端按需懒初始化（首次调用时从环境变量读取配置）。

    Examples
    --------
    ::

        embedder = Embedder()
        vector = await embedder.embed_query("退货政策是什么？")
    """

    def __init__(self, client: Optional[OpenAILikeEmbedding] = None) -> None:
        """
        Args:
            client: 预先创建的 OpenAILikeEmbedding 客户端；
                    为 ``None`` 时首次使用时从环境变量自动创建。
        """
        self._client = client

    def _ensure_client(self) -> OpenAILikeEmbedding:
        if self._client is None:
            logger.info("[Embedder] 从环境变量初始化 OpenAILikeEmbedding")
            self._client = OpenAILikeEmbedding.from_env()
        return self._client

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        批量向量化文本。

        Args:
            texts: 要向量化的文本列表。

        Returns:
            每条文本对应的向量（与索引维度一致）。
        """
        client = self._ensure_client()
        return await client._aget_text_embeddings(texts)

    async def embed_query(self, query: str) -> List[float]:
        """
        向量化单条查询文本。

        Args:
            query: 查询文本。

        Returns:
            查询向量。
        """
        vectors = await self.embed_texts([query])
        return vectors[0]
