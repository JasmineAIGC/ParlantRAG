"""
ParlantRetriever — 基于 Parlant SDK 的检索器。

将任意检索器（:class:`DenseRetriever` / :class:`KeywordRetriever`）封装，
提供 :meth:`retrieve` 独立使用接口和 :meth:`as_tool` Parlant SDK 集成接口。

与 ``FlexRAGRetriever`` 的区别：

* **无 FlexRAG / llama-index 依赖**：不加载 FAISS 持久化索引，
  改用 :class:`DenseRetriever` 在内存中直接构建向量索引，或使用
  :class:`KeywordRetriever` 做关键词检索，均无需安装 llama-index。
* **原生 Parlant SDK 集成**：:meth:`as_tool` 直接返回
  ``@p.tool`` 装饰的异步函数，可注册到 Parlant Agent 的 Guideline。
* **保持现有模型调用方式**：Embedding 仍走 ``model_client``
  （``OpenAILikeEmbedding`` / ``SentenceTransformerEmbedding``），
  与 :class:`DenseRetriever` 完全兼容。

依赖::

    pip install parlant          # Parlant SDK（仅 as_tool() 需要）
    pip install faiss-cpu        # DenseRetriever 模式所需
    pip install sentence-transformers  # 本地 Embedding 模式所需
"""
from typing import Annotated, Any

from rag_module.retrieval.types import Chunk, RetrievalResult


class ParlantRetriever:
    """
    基于 Parlant SDK 的检索器封装。

    接受任意实现了 ``retrieve(query, top_k)`` 异步接口的检索器作为底层实现，
    并通过 :meth:`as_tool` 将其暴露为 Parlant ``rag_search`` 工具。

    Usage（关键词检索，无额外依赖）::

        retriever = ParlantRetriever.from_keyword(documents=docs)
        result = await retriever.retrieve("查询词")

    Usage（向量检索，本地 sentence-transformers 模式）::

        retriever = ParlantRetriever.from_dense(documents=docs)
        await retriever.build_index()
        result = await retriever.retrieve("查询词")

    Usage（向量检索，API Embedding 模式）::

        from model_client.api.embedding import OpenAILikeEmbedding
        embed = OpenAILikeEmbedding.from_env()
        retriever = ParlantRetriever.from_dense(documents=docs, embed_model=embed)
        await retriever.build_index()

    Usage（注册到 Parlant Agent）::

        tool = retriever.as_tool()
        await agent.create_guideline(
            condition="用户提出了需要查询知识库才能回答的问题",
            action="使用 rag_search 工具检索相关内容。",
            tools=[tool],
        )
    """

    def __init__(self, base_retriever: Any) -> None:
        """
        Args:
            base_retriever: 任意检索器实例，需实现
                            ``retrieve(query: str, top_k: int)`` 异步方法。
                            通常为 :class:`~rag_module.retrieval.dense.DenseRetriever`
                            或 :class:`~rag_module.retrieval.keyword.KeywordRetriever`。
        """
        self._retriever = base_retriever

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def from_keyword(cls, documents: list[dict]) -> "ParlantRetriever":
        """
        使用 :class:`~rag_module.retrieval.keyword.KeywordRetriever` 创建实例。

        无需任何额外依赖，适合快速验证和小规模知识库。

        Args:
            documents: 文档列表，每条须含 ``id`` / ``content`` / ``source`` 字段。
        """
        from rag_module.retrieval.keyword import KeywordRetriever
        return cls(KeywordRetriever(documents))

    @classmethod
    def from_dense(
        cls,
        documents: list[dict],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embed_model: Any = None,
    ) -> "ParlantRetriever":
        """
        使用 :class:`~rag_module.retrieval.dense.DenseRetriever` 创建实例。

        创建后需调用 ``await retriever.build_index()`` 才可检索。

        Args:
            documents:   文档列表，每条须含 ``id`` / ``content`` / ``source`` 字段。
            model_name:  sentence-transformers 模型名称或本地路径；
                         ``embed_model`` 不为 ``None`` 时忽略此参数。
            embed_model: 可选的 API Embedding 模型实例（如 ``OpenAILikeEmbedding``）；
                         提供后使用 API 模式，优先级高于 ``model_name``。
        """
        from rag_module.retrieval.dense import DenseRetriever
        return cls(DenseRetriever(
            documents=documents,
            model_name=model_name,
            embed_model=embed_model,
        ))

    @classmethod
    def from_dense_env(cls, documents: list[dict]) -> "ParlantRetriever":
        """
        从环境变量读取 Embedding 配置，使用 API 模式创建向量检索器。

        读取的环境变量（优先级从高到低）：
        ``EMBEDDING_BASE_URL``、``EMBEDDING_API_KEY``、``EMBEDDING_MODEL`` 等，
        与 ``model_client.api.embedding.OpenAILikeEmbedding.from_env()`` 一致。

        创建后需调用 ``await retriever.build_index()`` 才可检索。

        Args:
            documents: 文档列表，每条须含 ``id`` / ``content`` / ``source`` 字段。
        """
        from rag_module.retrieval.dense import DenseRetriever
        return cls(DenseRetriever.from_env(documents=documents))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_index(self) -> None:
        """
        构建向量索引（仅 :class:`~rag_module.retrieval.dense.DenseRetriever` 需要）。

        关键词检索模式下此方法为空操作，可安全调用。
        """
        if hasattr(self._retriever, "build_index"):
            await self._retriever.build_index()

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        执行检索，返回 :class:`~rag_module.retrieval.types.RetrievalResult`。

        Args:
            query: 检索查询词。
            top_k: 返回片段数量上限。
        """
        return await self._retriever.retrieve(query=query, top_k=top_k)

    def as_tool(self) -> Any:
        """
        生成 Parlant ``rag_search`` 工具函数。

        返回一个 ``@p.tool`` 装饰的异步函数，可直接传入
        ``agent.create_guideline(tools=[...])`` 注册到 Parlant Agent。

        Returns:
            ``@p.tool`` 装饰的 ``rag_search`` 异步函数。

        Raises:
            ImportError: 若未安装 ``parlant`` SDK。
        """
        try:
            import parlant.sdk as p
        except ImportError:
            raise ImportError(
                "ParlantRetriever.as_tool() 需要安装 parlant SDK：\n"
                "pip install parlant"
            )

        retriever = self._retriever

        @p.tool
        async def rag_search(
            context: p.ToolContext,
            query: Annotated[
                str,
                p.ToolParameterOptions(
                    description="从用户问题中提炼的核心检索词"
                ),
            ],
            top_k: Annotated[
                int,
                p.ToolParameterOptions(
                    description="返回片段数量上限，默认 5"
                ),
            ] = 5,
        ) -> p.ToolResult:
            """从知识库检索与查询最相关的文档片段"""
            result = await retriever.retrieve(query=query, top_k=top_k)

            return p.ToolResult(
                data={
                    "query": result.query,
                    "retrieval_status": result.status,
                    "chunk_count": len(result.chunks),
                    "retrieved_chunks": [
                        {
                            "chunk_id": c.chunk_id,
                            "content": c.content,
                            "source": c.source,
                            "score": c.score,
                        }
                        for c in result.chunks
                    ],
                }
            )

        return rag_search
