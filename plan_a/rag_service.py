"""
RAG 服务层 — 封装检索管道，供 Scoped Retriever 和 RAG Tool 共用。

职责
----
1. 初始化 Embedder（使用 model_client.OpenAILikeEmbedding）
2. 加载 FAISS 向量索引（由知识库构建阶段预先生成）
3. 提供 :meth:`retrieve` 方法，返回结构化文档片段列表
4. 提供 :meth:`as_retriever_fn` 方法，返回符合 Parlant
   ``RetrieverFunction`` 签名的异步函数，可直接传给
   ``guideline.attach_retriever()``

设计说明
--------
- 不重复造轮子：底层复用 ``parlant_rag.retriever.Retriever``
  （已支持 FAISS 和 Parlant VectorDatabase 双模式）。
- 与 ``parlant_rag.adapters.ModelClientEmbedder`` 配合，
  实现 Parlant Embedder 接口 → model_client 调用链路。
- 配置通过 ``parlant_rag.config.RAGConfig`` 统一读取环境变量。
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any

import parlant.sdk as p

from parlant_rag.config import RAGConfig
from parlant_rag.retriever import Retriever
from parlant_rag.adapters import ModelClientEmbedder
from utils.logging import get_logger


logger = get_logger("plan_a.rag_service")


# ---------------------------------------------------------------------------
# 文档片段数据结构
# ---------------------------------------------------------------------------

class DocChunk:
    """检索到的文档片段（轻量数据容器，与 Parlant 序列化兼容）。"""

    __slots__ = ("content", "source", "score")

    SCORE_PRECISION = 4
    """序列化时 score 字段的小数位精度。"""

    def __init__(self, content: str, source: str, score: float) -> None:
        self.content = content
        self.source = source
        self.score = score

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，方便序列化给 Parlant RetrieverResult / ToolResult。"""
        return {
            "content": self.content,
            "source": self.source,
            "score": round(self.score, self.SCORE_PRECISION),
        }

    def __repr__(self) -> str:
        return f"DocChunk(source={self.source!r}, score={self.score:.3f})"


# ---------------------------------------------------------------------------
# RagService
# ---------------------------------------------------------------------------

class RagService:
    """
    RAG 检索服务。

    封装 Embedding + 向量检索全流程，暴露简洁的异步 ``retrieve`` 接口，
    同时提供 ``as_retriever_fn`` 工厂方法，生成可直接挂载到
    Parlant Guideline 上的 Scoped Retriever 函数。

    Examples
    --------
    ::

        # 纯检索模式（直接调用）
        service = RagService()
        await service.initialize()
        chunks = await service.retrieve("30天能退货吗？")

        # 作为 Parlant Scoped Retriever
        retriever_fn = service.as_retriever_fn()
        await guideline.attach_retriever(retriever_fn)
    """

    def __init__(self, config: Optional[RAGConfig] = None) -> None:
        """
        Args:
            config: RAG 配置；为 ``None`` 时自动从环境变量加载。
        """
        self._config = config or RAGConfig.from_env()
        self._retriever: Optional[Retriever] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """初始化 Embedder 和 FAISS 索引（幂等，可多次调用）。"""
        if self._initialized:
            return

        logger.info("[RagService] 开始初始化")

        embedder = ModelClientEmbedder()
        self._retriever = Retriever(
            embedder=embedder,
            persist_dir=self._config.knowledge_base_path,
            collection_name=self._config.vector_db_collection,
            top_k=self._config.top_k,
        )
        await self._retriever.initialize()

        self._initialized = True
        logger.info("[RagService] 初始化完成")

    # ------------------------------------------------------------------
    # 核心接口
    # ------------------------------------------------------------------

    async def retrieve(self, query: str) -> List[DocChunk]:
        """
        检索与 ``query`` 最相关的文档片段。

        Args:
            query: 用户查询文本。

        Returns:
            相关文档片段列表（按相似度降序），空列表表示未命中。
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"[RagService] 检索 query={query!r}")

        result = await self._retriever.retrieve(query)

        chunks = [
            DocChunk(
                content=c.content,
                source=c.source,
                score=c.score,
            )
            for c in result.chunks
        ]

        logger.info(f"[RagService] 命中 {len(chunks)} 个片段")
        return chunks

    # ------------------------------------------------------------------
    # Parlant 集成工厂
    # ------------------------------------------------------------------

    def as_retriever_fn(self):
        """
        返回符合 Parlant ``RetrieverFunction`` 签名的异步函数。

        返回值可直接传入 ``guideline.attach_retriever()`` 或
        ``agent.attach_retriever()``，使 Guideline 匹配时自动执行
        知识库检索并将结果注入 Agent 的对话上下文。

        Returns
        -------
        Callable[[RetrieverContext], Awaitable[RetrieverResult | None]]
            Parlant Scoped Retriever 函数。

        Examples
        --------
        ::

            from plan_a import RagService
            import parlant.sdk as p

            rag = RagService()
            await rag.initialize()

            async with p.Server() as server:
                agent = await server.create_agent(...)
                gl = await agent.create_guideline(
                    condition="用户询问退货政策",
                    action="检索知识库后回答",
                )
                await gl.attach_retriever(rag.as_retriever_fn())
        """
        async def _retriever_fn(
            context: p.RetrieverContext,
        ) -> p.RetrieverResult | None:
            """Parlant Scoped Retriever：从知识库自动检索与当前对话相关的内容。"""
            last_msg = context.interaction.last_customer_message
            if not last_msg:
                logger.debug("[RagService] 未找到用户消息，跳过检索")
                return None

            query = last_msg.content
            chunks = await self.retrieve(query)

            if not chunks:
                logger.info("[RagService] Scoped Retriever：未检索到相关内容")
                return None

            # 将检索结果格式化为可读文本，作为 data 注入 Agent 上下文
            formatted_text = "\n\n".join(
                f"[来源: {c.source}]\n{c.content}" for c in chunks
            )

            logger.info(
                f"[RagService] Scoped Retriever：返回 {len(chunks)} 个片段"
            )
            return p.RetrieverResult(
                data=formatted_text,
                metadata={"chunk_count": len(chunks), "query": query},
            )

        return _retriever_fn
