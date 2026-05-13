"""
FlexRAGRetriever — 加载 FlexRAG 构建的 FAISS 向量知识库。

知识库调用方式参考 https://github.com/Xavier-123/FlexRAG 项目中的以下实现：

* **索引加载** — 参考 ``MultiVectorRetriever._load_index()``
  （``flexrag/components/retrieval/multi_vector_retriever.py``）：
  读取 ``faiss_index.bin``，通过 ``FaissVectorStore + StorageContext``
  恢复 ``VectorStoreIndex``。

* **混合检索算法** — 参考 ``HybridRetriever.retrieve()``
  （``flexrag/components/retrieval/retrieval_opt.py``）：

  1. 按权重缩放各路（向量 / BM25）召回分数。
  2. 以 ``metadata["id"]``（优先）或文档文本（兜底）为键去重。
  3. 同一文档多路命中时，保留分数更高的那条（max-score 策略）。
  4. 按分数降序排列，截断至 ``top_k``。

主要组件对应关系：

=============  ===============================================
本模块            FlexRAG 原始位置
=============  ===============================================
OpenAILikeEmbedding   flexrag/components/retrieval/base.py
                      （实现见 rag_module/retrieval/embedding.py）
FlexRAGRetriever      flexrag/components/retrieval/multi_vector_retriever.py
                      (MultiVectorRetriever._load_index + retrieve)
                  +   flexrag/components/retrieval/retrieval_opt.py
                      (HybridRetriever.retrieve — 去重 & 融合算法)
=============  ===============================================

依赖::

    pip install llama-index llama-index-vector-stores-faiss faiss-cpu
    # BM25 混合（可选）：
    pip install llama-index-retrievers-bm25 rank_bm25 jieba
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except ImportError:
    pass

from rag_module.base import BaseRetriever, Chunk, RetrievalResult
from rag_module.retrieval.embedding import OpenAILikeEmbedding, SentenceTransformerEmbedding

logger = logging.getLogger(__name__)

_FAISS_FILE = "faiss_index.bin"


def _first_env_local(keys: Sequence[str], default: Optional[str] = None) -> Optional[str]:
    """按顺序读取环境变量，返回首个非空值。"""
    for key in keys:
        if value := os.getenv(key):
            return value
    return default


# ---------------------------------------------------------------------------
# FlexRAGRetriever
# ---------------------------------------------------------------------------

class FlexRAGRetriever(BaseRetriever):
    """
    读取由 FlexRAG ``MultiVectorRetriever`` 持久化的目录，执行向量（+可选 BM25）检索。

    索引加载方式与 FlexRAG ``MultiVectorRetriever._load_index()`` 完全一致：

    1. 读取 ``{persist_dir}/faiss_index.bin``
    2. 构造 ``FaissVectorStore`` + ``StorageContext``
    3. 调用 ``load_index_from_storage()`` 恢复 ``VectorStoreIndex``
    4. 通过 ``index.as_retriever()`` 执行相似度检索

    构造时立即加载索引（同步），检索时异步执行。

    Usage::

        retriever = FlexRAGRetriever(
            persist_dir="./knowledge_base",
            embedding_model_path="/path/to/models/BAAI/bge-large-zh-v1.5",
            embedding_model="bge-large-zh-v1.5",
            top_k=5,
        )
        # Or load everything from .env / environment variables:
        retriever = FlexRAGRetriever.from_env()
        result = await retriever.retrieve("查询词")
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_base_url: str = "http://127.0.0.1:8018/v1",
        embedding_api_key: Optional[str] = None,
        embedding_access_key: Optional[str] = None,
        embedding_model: str = "bge-large-zh-v1.5",
        embedding_model_path: Optional[str] = None,
        top_k: int = 5,
        bm25_weight: float = 0.0,
    ) -> None:
        """
        Args:
            persist_dir:            FlexRAG 知识库根目录（包含 ``faiss_index.bin`` 等文件）。
            embedding_base_url:     保留参数（不再使用），仅用于向后兼容。
            embedding_api_key:      保留参数（不再使用），仅用于向后兼容。
            embedding_access_key:   保留参数（不再使用），仅用于向后兼容。
            embedding_model:        Embedding 模型名称，须与构建知识库时完全一致。
            embedding_model_path:   本地 sentence-transformers 模型路径（**必填**）。
            top_k:                  默认返回的片段数量。
            bm25_weight:            BM25 混合权重（0~1）。0.0 表示纯向量检索。
                                    当 ``persist_dir/bm25_index/`` 存在且此值 > 0 时
                                    自动启用混合检索。
        """
        self._persist_dir = persist_dir
        self._embedding_base_url = embedding_base_url
        self._embedding_api_key = embedding_api_key
        self._embedding_access_key = embedding_access_key
        self._embedding_model = embedding_model
        self._embedding_model_path = embedding_model_path
        self._top_k = top_k
        self._bm25_weight = bm25_weight

        self._index: Any = None
        self._embed_model: Any = None
        self._bm25_retriever: Any = None

        self._load_index()

    @classmethod
    def from_env(cls, persist_dir: Optional[str] = None) -> "FlexRAGRetriever":
        """
        从环境变量（或 ``.env`` 文件）加载 Embedding 配置并创建检索器实例。

        支持两种 Embedding 模式，优先级：**本地 > API**：

        **方式一：本地 sentence-transformers（EMBEDDING_MODEL_PATH 已设置时生效）**

        ========================  =========================================
        参数                       读取的环境变量
        ========================  =========================================
        persist_dir               KNOWLEDGE_PERSIST_DIR / KNOWLEDGE_BASE_DIR
        embedding_model_path      EMBEDDING_MODEL_PATH（**必须设置**）
        embedding_model           EMBEDDING_MODEL（可选，默认取路径末段目录名）
        top_k                     TOP_K_RETRIEVAL / TOP_K
        bm25_weight               BM25_WEIGHT
        ========================  =========================================

        **方式二：API 服务 / vLLM Embedding（EMBEDDING_BASE_URL 已设置时生效）**

        ========================  =========================================
        参数                       读取的环境变量
        ========================  =========================================
        persist_dir               KNOWLEDGE_PERSIST_DIR / KNOWLEDGE_BASE_DIR
        embedding_base_url        EMBEDDING_BASE_URL（**必须设置**）
        embedding_api_key         EMBEDDING_API_KEY / QWEN3_API_KEY / LLM_API_KEY
        embedding_access_key      EMBEDDING_ACCESS_KEY / QWEN3_ACCESS_KEY /
                                  CONTEXTMIND_ACCESS_KEY / LLM_ACCESS_KEY / ACCESS_KEY
        embedding_model           EMBEDDING_MODEL / QWEN3_MODEL / LLM_MODEL
        top_k                     TOP_K_RETRIEVAL / TOP_K
        bm25_weight               BM25_WEIGHT
        ========================  =========================================

        两者均未设置时抛出 :class:`ValueError`。
        """
        pd = persist_dir or _first_env_local(
            ["KNOWLEDGE_PERSIST_DIR", "KNOWLEDGE_BASE_DIR"],
            default="./knowledge_base/knowledge_persist_dir/hotpotqa",
        )

        top_k = int(_first_env_local(["TOP_K_RETRIEVAL", "TOP_K"]) or "5")
        bm25_weight = float(_first_env_local(["BM25_WEIGHT"]) or "0.0")

        # --- 本地模式（优先）---
        model_path = os.getenv("EMBEDDING_MODEL_PATH")
        if model_path:
            model = os.getenv("EMBEDDING_MODEL") or os.path.basename(model_path.rstrip("/\\"))
            return cls(
                persist_dir=pd,
                embedding_model=model,
                embedding_model_path=model_path,
                top_k=top_k,
                bm25_weight=bm25_weight,
            )

        # --- API 模式（EMBEDDING_BASE_URL 已配置）---
        base_url = os.getenv("EMBEDDING_BASE_URL")
        if base_url:
            api_key = _first_env_local(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"])
            access_key = _first_env_local(
                [
                    "EMBEDDING_ACCESS_KEY",
                    "QWEN3_ACCESS_KEY",
                    "CONTEXTMIND_ACCESS_KEY",
                    "LLM_ACCESS_KEY",
                    "ACCESS_KEY",
                ]
            )
            model = _first_env_local(
                ["EMBEDDING_MODEL", "QWEN3_MODEL", "LLM_MODEL"],
                default="bge-large-zh-v1.5",
            )
            return cls(
                persist_dir=pd,
                embedding_base_url=base_url,
                embedding_api_key=api_key,
                embedding_access_key=access_key,
                embedding_model=model,
                embedding_model_path=None,
                top_k=top_k,
                bm25_weight=bm25_weight,
            )

        raise ValueError(
            "未检测到 Embedding 配置，请在 .env 文件中设置以下其中一组：\n"
            "\n"
            "  方式一（本地 sentence-transformers）：\n"
            "    EMBEDDING_MODEL_PATH=/path/to/your/models/BAAI/bge-large-zh-v1.5\n"
            "    EMBEDDING_MODEL=bge-large-zh-v1.5  # 可选\n"
            "\n"
            "  方式二（API 服务 / vLLM）：\n"
            "    EMBEDDING_BASE_URL=http://your-host:port/v1\n"
            "    EMBEDDING_API_KEY=your-api-key\n"
            "    EMBEDDING_MODEL=your-embedding-model-name\n"
        )

    @staticmethod
    def get_index_dim(persist_dir: str) -> int:
        """
        读取 FAISS 索引的向量维度，**不加载任何 Embedding 模型**。

        在配置 ``EMBEDDING_MODEL_PATH`` 之前调用此方法，可确认知识库需要的向量维度，
        从而选用输出维度一致的 Embedding 模型。

        Args:
            persist_dir: FlexRAG 知识库根目录（包含 ``faiss_index.bin``）。

        Returns:
            FAISS 索引的向量维度（整数）。

        Raises:
            FileNotFoundError: 目录下找不到 ``faiss_index.bin``。
            ImportError:       ``faiss`` 未安装。

        Example::

            dim = FlexRAGRetriever.get_index_dim("./knowledge_base/knowledge_persist_dir/hotpotqa")
            print(f"知识库需要维度为 {dim} 的 Embedding 模型")
        """
        try:
            import faiss as _faiss
        except ImportError:
            raise ImportError(
                "读取 FAISS 索引维度需要安装 faiss：pip install faiss-cpu"
            )
        faiss_path = os.path.join(persist_dir, _FAISS_FILE)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"未找到 FAISS 索引文件：{faiss_path}\n"
                f"请确认 persist_dir 路径正确，且知识库已构建完成。"
            )
        return _faiss.read_index(faiss_path).d

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def persist_dir(self) -> str:
        """知识库根目录路径。"""
        return self._persist_dir

    @property
    def embedding_model(self) -> str:
        """Embedding 模型名称。"""
        return self._embedding_model

    @property
    def embedding_model_path(self) -> Optional[str]:
        """本地 Embedding 模型路径（本地模式时有值，API 模式时为 None）。"""
        return self._embedding_model_path

    @property
    def embedding_base_url(self) -> str:
        """Embedding 服务的 base URL（API 模式时有效）。"""
        return self._embedding_base_url

    @property
    def embedding_mode(self) -> str:
        """Embedding 模式：``'local'`` 或 ``'api'``。"""
        return "local" if self._embedding_model_path else "api"

    @property
    def doc_count(self) -> int:
        """已加载的文档片段总数；索引未加载时返回 0。"""
        if self._index is None:
            return 0
        try:
            return len(self._index.docstore.docs)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """
        加载 FAISS 持久化索引（以及可选的 BM25 索引）。

        实现参考 FlexRAG ``MultiVectorRetriever._load_index()``：
        使用 ``SentenceTransformerEmbedding`` 在本地加载 Embedding 模型
        （与 ``temp/embedding.py`` 中的 ``EmbeddingClient`` 本地模式一致），
        通过 ``FaissVectorStore + StorageContext`` 恢复 ``VectorStoreIndex``。

        加载过程中会主动探测 Embedding 模型的输出维度，并与 FAISS 索引维度比对。
        若两者不一致，立即抛出清晰的 ``ValueError``，而非等到检索时因 FAISS
        ``assert d == self.d`` 抛出难以理解的 ``AssertionError``。
        """
        try:
            import faiss as _faiss
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError:
            raise ImportError(
                "FlexRAGRetriever 需要安装：\n"
                "pip install llama-index llama-index-vector-stores-faiss faiss-cpu httpx"
            )

    def _load_index(self) -> None:
        """
        加载 FAISS 持久化索引（以及可选的 BM25 索引）。

        支持两种 Embedding 模式：

        * **本地模式**（``embedding_model_path`` 已设置）：使用
          ``SentenceTransformerEmbedding`` 在本地推理。
        * **API 模式**（``embedding_model_path`` 未设置，但 ``embedding_base_url``
          已设置）：使用 ``OpenAILikeEmbedding`` 调用外部 vLLM / OpenAI 兼容服务。

        加载过程中会主动探测 Embedding 模型的输出维度，并与 FAISS 索引维度比对。
        若两者不一致，立即抛出清晰的 ``ValueError``，而非等到检索时因 FAISS
        ``assert d == self.d`` 抛出难以理解的 ``AssertionError``。
        """
        try:
            import faiss as _faiss
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError:
            raise ImportError(
                "FlexRAGRetriever 需要安装：\n"
                "pip install llama-index llama-index-vector-stores-faiss faiss-cpu httpx"
            )

        faiss_path = os.path.join(self._persist_dir, _FAISS_FILE)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"未找到 FAISS 索引文件：{faiss_path}\n"
                "请先执行 FlexRAG scripts/build_knowledge_base.py 构建知识库，"
                "或将 KNOWLEDGE_PERSIST_DIR 设置为已有知识库的正确目录（含 faiss_index.bin）。"
            )

        # --- 选择 Embedding 后端 ---
        if self._embedding_model_path:
            # 本地 sentence-transformers 模式
            self._embed_model = SentenceTransformerEmbedding(
                model_path=self._embedding_model_path,
                model_name=self._embedding_model,
            )
            logger.info(
                "[FlexRAGRetriever] 使用本地 Embedding 模型  model=%s  path=%s",
                self._embedding_model, self._embedding_model_path,
            )
        elif self._embedding_base_url:
            # API 模式：OpenAI 兼容 Embedding 服务
            self._embed_model = OpenAILikeEmbedding(
                base_url=self._embedding_base_url,
                model_name=self._embedding_model,
                api_key=self._embedding_api_key,
                access_key=self._embedding_access_key,
            )
            logger.info(
                "[FlexRAGRetriever] 使用 API Embedding 服务  model=%s  url=%s",
                self._embedding_model, self._embedding_base_url,
            )
        else:
            raise ValueError(
                "未配置 Embedding 后端。\n"
                "请设置以下其中一组环境变量：\n"
                "  本地模式：EMBEDDING_MODEL_PATH=/path/to/model\n"
                "  API  模式：EMBEDDING_BASE_URL=http://host:port/v1  EMBEDDING_MODEL=model-name\n"
                "或直接使用 FlexRAGRetriever.from_env() 自动读取配置。"
            )
        Settings.embed_model = self._embed_model

        # --- 主动维度校验（参考 MultiVectorRetriever._detect_embedding_dim）---
        # 在加载 FAISS 索引后立即比对维度，确保 Embedding 模型与索引完全兼容。
        # 若不一致，此处抛出清晰 ValueError，而非等到 retrieve() 时
        # 遇到 FAISS 内部 "assert d == self.d" 的 AssertionError。
        faiss_index = _faiss.read_index(faiss_path)
        faiss_dim: int = faiss_index.d

        try:
            probe_vec = self._embed_model.get_text_embedding("维度探测")
            embed_dim = len(probe_vec)
        except Exception as probe_exc:
            mode_desc = (
                f"路径：{self._embedding_model_path}"
                if self._embedding_model_path
                else f"URL：{self._embedding_base_url}"
            )
            raise RuntimeError(
                f"Embedding 模型维度探测失败（模型：{self._embedding_model}，"
                f"{mode_desc}）。\n"
                f"原始错误：{probe_exc}"
            ) from probe_exc

        if embed_dim != faiss_dim:
            mode_desc = (
                f"  当前模型路径：{self._embedding_model_path}\n"
                if self._embedding_model_path
                else f"  当前 API URL：{self._embedding_base_url}\n"
            )
            raise ValueError(
                f"Embedding 维度与 FAISS 索引不匹配！\n"
                f"  FAISS 索引维度：{faiss_dim}\n"
                f"  当前模型输出维度：{embed_dim}\n"
                f"  当前模型名称：{self._embedding_model}\n"
                f"{mode_desc}"
                f"请换用输出维度为 {faiss_dim} 的 Embedding 模型（与构建知识库时完全一致），\n"
                f"或在配置前先调用 FlexRAGRetriever.get_index_dim(persist_dir) 查询知识库所需维度。"
            )

        logger.info(
            "[FlexRAGRetriever] Embedding 维度校验通过  embed_dim=%d  faiss_dim=%d",
            embed_dim, faiss_dim,
        )

        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self._persist_dir,
        )
        self._index = load_index_from_storage(storage_context)

        # 可选：加载 BM25 索引
        bm25_path = os.path.join(self._persist_dir, "bm25_index")
        if self._bm25_weight > 0 and os.path.isdir(bm25_path):
            self._load_bm25(bm25_path)

    def _load_bm25(self, bm25_path: str) -> None:
        """加载 BM25 索引（可选）；失败时自动回退到纯向量检索。"""
        try:
            from llama_index.retrievers.bm25 import BM25Retriever as _BM25Retriever
            self._bm25_retriever = _BM25Retriever.from_persist_dir(bm25_path)
        except Exception:
            logger.warning(
                "无法加载 BM25 索引（路径：%s），回退到纯向量检索。",
                bm25_path,
                exc_info=True,
            )
            self._bm25_weight = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """
        执行向量检索（可选 BM25 混合），返回 :class:`~rag_module.base.RetrievalResult`。

        算法参考 FlexRAG ``retrieval_opt.HybridRetriever.retrieve()``
        （``third_party/FlexRAG/flexrag/components/retrieval/retrieval_opt.py``）：

        1. 通过 llama-index 的异步 ``aretrieve()`` 执行向量检索（避免嵌套事件循环）。
        2. 可选地调用 BM25 检索，按权重缩放各路分数。
        3. 以 ``metadata["id"]``（优先）或文档文本（兜底）为键去重——
           同一文档出现多次时，保留分数更高的那条（max-score 策略，与 FlexRAG 一致）。
        4. 按分数降序排列，截断至 ``top_k``。

        Args:
            query:  检索查询词。
            top_k:  返回片段数量，不传则使用构造时的 ``top_k``。

        Returns:
            :class:`~rag_module.base.RetrievalResult`
        """
        k = top_k if top_k is not None else self._top_k
        dense_w = 1.0 - self._bm25_weight

        logger.info("[FlexRAGRetriever] 检索开始  query=%r  top_k=%d", query, k)

        # --- 向量检索 ---
        # 使用 llama-index 的异步 aretrieve()：它直接走 _aget_query_embedding() 路径，
        # 不会在事件循环内部再调用 asyncio.get_event_loop().run_until_complete()，
        # 从而避免 "This event loop is already running" 的嵌套事件循环冲突。
        dense_retriever = self._index.as_retriever(similarity_top_k=k)
        try:
            dense_nodes = await dense_retriever.aretrieve(query)
        except AssertionError as exc:
            # FAISS 内部 `assert d == self.d` 失败：查询向量维度与索引不一致。
            # 正常情况下 _load_index() 的主动维度校验会提前捕获这个问题；
            # 这里作为安全兜底，转换为可读的 ValueError。
            logger.exception("[FlexRAGRetriever] 向量检索失败（维度不匹配）  query=%r", query)
            raise ValueError(
                f"FAISS 向量维度断言失败（assert d == self.d）。\n"
                f"查询时的 Embedding 模型输出维度与知识库索引维度不一致。\n"
                f"请确认 EMBEDDING_MODEL_PATH 所指向的模型与构建知识库时使用的模型完全一致。\n"
                f"可通过 FlexRAGRetriever.get_index_dim(persist_dir) 查询知识库所需的向量维度。"
            ) from exc
        except Exception as exc:
            exc_str = str(exc)
            # httpx / httpcore 连接错误：Embedding 服务不可达
            if "ConnectError" in type(exc).__name__ or "ConnectError" in exc_str:
                mode = "本地" if self._embedding_model_path else f"API（{self._embedding_base_url}）"
                raise ConnectionError(
                    f"无法连接到 Embedding 服务，请检查配置。\n"
                    f"  当前模式：{mode}\n"
                    f"  模型：{self._embedding_model}\n"
                    f"  提示：\n"
                    f"    - 如需使用本地模型，请在 .env 中设置 EMBEDDING_MODEL_PATH=/path/to/model\n"
                    f"    - 如需使用 API 服务，请将 EMBEDDING_BASE_URL 设为可访问的地址\n"
                    f"      （避免使用 QWEN3_BASE_URL，那是 LLM 服务地址）"
                ) from exc
            logger.exception("[FlexRAGRetriever] 向量检索失败  query=%r", query)
            raise

        logger.info("[FlexRAGRetriever] 向量检索返回 %d 个节点", len(dense_nodes))

        # --- 多路召回结果收集 ---
        # 每条结果：(node, weighted_score)
        all_nodes: list[tuple[Any, float]] = []
        for node in dense_nodes:
            all_nodes.append((node, (node.score or 0.0) * dense_w))

        if self._bm25_retriever is not None and self._bm25_weight > 0:
            bm25_nodes = await self._bm25_retriever.aretrieve(query)
            logger.info("[FlexRAGRetriever] BM25 检索返回 %d 个节点", len(bm25_nodes))
            for node in bm25_nodes:
                all_nodes.append((node, (node.score or 0.0) * self._bm25_weight))

        logger.info("[FlexRAGRetriever] 去重前共 %d 条结果", len(all_nodes))

        # --- 去重（参考 FlexRAG HybridRetriever 的策略）---
        # 以 metadata["id"] 为首选去重键（推荐），回退到文档文本（兜底）。
        # 同一文档出现多次时保留分数更高的条目（max-score 策略）。
        dedup: dict[str, tuple[Any, float]] = {}
        for node, score in all_nodes:
            meta = node.metadata or {}
            key: str = meta.get("id") or node.get_content()
            if key not in dedup:
                dedup[key] = (node, score)
            elif score > dedup[key][1]:
                dedup[key] = (node, score)

        logger.info("[FlexRAGRetriever] 去重后共 %d 个唯一文档", len(dedup))

        # --- 排序 & 截断 & 封装为 Chunk ---
        sorted_items = sorted(dedup.values(), key=lambda x: x[1], reverse=True)[:k]

        chunks: list[Chunk] = []
        for idx, (node, score) in enumerate(sorted_items):
            meta = node.metadata or {}
            source = (
                meta.get("file_name")
                or meta.get("title")
                or meta.get("source")
                or "unknown"
            )
            chunks.append(
                Chunk(
                    chunk_id=str(meta.get("chunk_id", idx)),
                    content=node.get_content(),
                    source=source,
                    score=round(score, 4),
                    metadata=meta,
                )
            )

        logger.info("[FlexRAGRetriever] 最终返回 %d 个片段", len(chunks))

        return RetrievalResult(
            query=query,
            chunks=chunks,
            status="success" if chunks else "no_results",
        )
