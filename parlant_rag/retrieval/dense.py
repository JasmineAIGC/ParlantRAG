"""
DenseRetriever — 本地向量检索器（sentence-transformers + FAISS）。

适合生产环境中不依赖外部 Embedding 服务的场景。
也支持通过 ``embed_model`` 参数传入 ``OpenAILikeEmbedding``，
使用 OpenAI 兼容 API 提供 embedding，适配已部署的 Embedding 服务。

依赖::

    # 本地模式（默认）：
    pip install sentence-transformers faiss-cpu
    # API 模式（embed_model 参数）：
    pip install llama-index openai faiss-cpu
"""
from __future__ import annotations

import asyncio
from typing import Any, List, Optional

from parlant_rag.retrieval.types import Chunk, RetrievalResult


class DenseRetriever:
    """
    基于向量编码 + FAISS 内积索引的向量检索器。

    支持两种 Embedding 后端：

    * **sentence-transformers 本地模式**（默认）：``model_name`` 指定模型名或路径，
      无需网络请求。
    * **OpenAI 兼容 API 模式**：传入 ``embed_model``（``OpenAILikeEmbedding`` 实例），
      通过 vLLM / OpenAI 兼容接口完成向量编码，与 ``FlexRAGRetriever`` 的 Embedding
      来源保持一致，适合已有在线 Embedding 服务的生产环境。

    调用 :meth:`build_index` 一次后，后续的 :meth:`retrieve` 均为纯内存操作。

    Usage::

        # 本地 sentence-transformers 模式
        retriever = DenseRetriever(
            documents=docs,
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
        )
        await retriever.build_index()

        # API 模式（从环境变量读取配置）
        retriever = DenseRetriever.from_env(documents=docs)
        await retriever.build_index()

        result = await retriever.retrieve("查询词", top_k=5)
    """

    def __init__(
        self,
        documents: list[dict],
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embed_model: Any = None,
    ) -> None:
        """
        Args:
            documents:   文档列表，每条必须包含 ``id`` / ``content`` / ``source`` 字段。
            model_name:  sentence-transformers 模型名称（或本地路径）。
                         仅在 ``embed_model`` 为 ``None`` 时使用。
            embed_model: 可选的 Embedding 模型实例（如 ``OpenAILikeEmbedding``）。
                         提供后将使用 API 模式，忽略 ``model_name``。
        """
        self._docs = documents
        self._model_name = model_name
        self._embed_model = embed_model
        self._index: Any = None
        self._model: Any = None  # sentence-transformers model (local mode only)

    @classmethod
    def from_env(cls, documents: list[dict]) -> "DenseRetriever":
        """
        从环境变量创建使用 ``OpenAILikeEmbedding`` API 模式的检索器。

        读取的环境变量与 ``FlexRAGRetriever.from_env()`` 完全一致（``EMBEDDING_*``
        优先，其次 ``QWEN3_*``，最后 ``LLM_*``）。

        Args:
            documents: 文档列表，每条必须包含 ``id`` / ``content`` / ``source`` 字段。
        """
        from model_client.api.embedding import OpenAILikeEmbedding
        embed_model = OpenAILikeEmbedding.from_env()
        return cls(documents=documents, embed_model=embed_model)

    async def build_index(self) -> None:
        """构建向量索引。程序启动时调用一次即可。"""
        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError(
                "DenseRetriever 需要安装额外依赖：\n"
                "pip install faiss-cpu"
            )

        texts = [doc["content"] for doc in self._docs]

        if self._embed_model is not None:
            # --- API 模式：使用 OpenAILikeEmbedding ---
            batch_size: int = getattr(self._embed_model, "embed_batch_size", 5)
            all_embeddings: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                batch_embs = await self._embed_model._aget_text_embeddings(batch)
                all_embeddings.extend(batch_embs)
            embeddings = np.array(all_embeddings, dtype=np.float32)
        else:
            # --- 本地模式：使用 sentence-transformers ---
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "DenseRetriever 本地模式需要安装额外依赖：\n"
                    "pip install sentence-transformers"
                )

            def _build() -> Any:
                model = SentenceTransformer(self._model_name)
                self._model = model
                return model.encode(texts, normalize_embeddings=True)

            # sentence_transformers.encode() 是阻塞调用，通过 to_thread 避免阻塞事件循环
            embeddings = await asyncio.to_thread(_build)

        # 归一化，使内积等价于余弦相似度
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = (embeddings / norms).astype(np.float32)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

    async def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        """
        Args:
            query:  检索查询词
            top_k:  返回片段数量上限

        Raises:
            RuntimeError: 若尚未调用 :meth:`build_index`。
        """
        if self._index is None:
            raise RuntimeError(
                "DenseRetriever 尚未构建索引，请先调用 await retriever.build_index()"
            )

        import numpy as np

        if self._embed_model is not None:
            # --- API 模式 ---
            query_vec_list = await self._embed_model._aget_query_embedding(query)
            query_vec = np.array([query_vec_list], dtype=np.float32)
        else:
            # --- 本地模式 ---
            if self._model is None:
                raise RuntimeError(
                    "DenseRetriever 尚未构建索引，请先调用 await retriever.build_index()"
                )
            model = self._model
            query_vec = await asyncio.to_thread(
                lambda: model.encode([query], normalize_embeddings=True).astype(np.float32)
            )

        # 归一化查询向量
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        scores, indices = self._index.search(query_vec, top_k)

        chunks = [
            Chunk(
                chunk_id=self._docs[idx]["id"],
                content=self._docs[idx]["content"],
                source=self._docs[idx]["source"],
                score=round(float(scores[0][i]), 4),
                metadata=self._docs[idx].get("metadata", {}),
            )
            for i, idx in enumerate(indices[0])
            if idx >= 0 and scores[0][i] > 0
        ]

        return RetrievalResult(
            query=query,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )
