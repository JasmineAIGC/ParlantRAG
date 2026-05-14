"""
rag_pipeline.py — RAG 检索管道（核心服务层）。

对应架构图中的 ``RAG[RAG 服务/Pipeline]`` 模块。

职责：
  将 Embedding 组件和 VectorStore 组件串联，
  接收自然语言查询，返回结构化的文档片段列表。

  Scoped Retriever 和 RAG Tool 均通过此 Pipeline 进行检索，
  二者共享同一 Pipeline 实例，避免重复初始化。

检索流程::

    query (str)
        │
        ▼ Embedder.embed_query()
    query_vector (List[float])
        │
        ▼ VectorStore.search()
    List[DocChunk]  ← 返回给调用方

本文件不依赖 parlant_rag/ 或其他 RAG 实现模块。
"""
from __future__ import annotations

from typing import List, Optional

from plan_a.config import PlanAConfig
from plan_a.types import DocChunk
from plan_a.embedding import Embedder
from plan_a.vector_store import VectorStore
from utils.logging import get_logger


logger = get_logger("plan_a.rag_pipeline")


class RAGPipeline:
    """
    RAG 检索管道。

    初始化后即可通过 :meth:`retrieve` 对外提供检索服务。
    Scoped Retriever（:mod:`plan_a.scoped_retriever`）和
    RAG Tool（:mod:`plan_a.rag_tool`）均使用同一 Pipeline 实例。

    Examples
    --------
    ::

        config = PlanAConfig.from_env()
        pipeline = RAGPipeline(config)
        await pipeline.initialize()

        chunks = await pipeline.retrieve("退货政策是什么？")
        for c in chunks:
            print(c.source, c.score, c.content[:80])
    """

    def __init__(
        self,
        config: Optional[PlanAConfig] = None,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        """
        Args:
            config:        RAG 配置；``None`` 时从环境变量自动加载。
            embedder:      Embedding 组件；``None`` 时使用默认 :class:`~plan_a.embedding.Embedder`。
            vector_store:  VectorStore 组件；``None`` 时使用默认 :class:`~plan_a.vector_store.VectorStore`。
        """
        self._config = config or PlanAConfig.from_env()
        self._embedder = embedder or Embedder()
        self._vector_store = vector_store or VectorStore()
        self._initialized = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """
        初始化管道（幂等，可多次调用）。

        步骤：
          1. 加载 FAISS 索引和文档库（VectorStore）
          2. Embedder 客户端在首次 embed_query() 时懒初始化

        Raises:
            FileNotFoundError: 若知识库目录或索引文件不存在。
            ImportError: 若未安装 ``faiss-cpu`` / ``faiss-gpu``。
        """
        if self._initialized:
            return

        logger.info("[RAGPipeline] 初始化开始")
        await self._vector_store.initialize(self._config.knowledge_base_path)
        self._initialized = True
        logger.info("[RAGPipeline] 初始化完成")

    # ------------------------------------------------------------------
    # 检索接口
    # ------------------------------------------------------------------

    async def retrieve(self, query: str) -> List[DocChunk]:
        """
        检索与查询最相关的文档片段。

        若管道尚未初始化，自动调用 :meth:`initialize`。

        Args:
            query: 用户查询文本。

        Returns:
            按相似度降序排列的 :class:`~plan_a.types.DocChunk` 列表；
            空列表表示知识库中未找到相关内容。
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"[RAGPipeline] 检索  query={query!r}")

        query_vector = await self._embedder.embed_query(query)
        chunks = self._vector_store.search(query_vector, self._config.top_k)

        logger.info(f"[RAGPipeline] 命中 {len(chunks)} 个片段")
        return chunks
