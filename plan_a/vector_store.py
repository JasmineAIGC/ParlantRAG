"""
vector_store.py — 向量数据库层（VDB）。

对应架构图中的 ``VDB[(向量数据库)]`` 模块。

职责：
  1. 加载预先构建好的 FAISS 索引和文档库（不负责索引构建）
  2. 接受查询向量，执行近邻搜索（FAISS）
  3. 返回 :class:`~plan_a.types.DocChunk` 列表

预期的知识库目录结构
-------------------
::

    {persist_dir}/
        faiss_index           ← faiss.write_index() 写入的文件
        document_store.json   ← {"doc_id": {"content": str, "source": str, "metadata": {}}, ...}

若目录不存在或索引文件缺失，抛出 ``FileNotFoundError``。

本文件不依赖 parlant_rag/ 或其他 RAG 实现模块。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from plan_a.types import DocChunk
from utils.logging import get_logger


logger = get_logger("plan_a.vector_store")


class VectorStore:
    """
    本地 FAISS 向量库。

    加载由外部工具（如 FlexRAG）预先构建的 FAISS 索引，
    并提供同步的向量近邻搜索接口。

    Examples
    --------
    ::

        store = VectorStore()
        await store.initialize("./knowledge_base/hotpotqa")
        chunks = store.search(query_vector, top_k=5)
    """

    _INDEX_FILENAME = "faiss_index"
    _STORE_FILENAME = "document_store.json"

    def __init__(self) -> None:
        self._faiss_index = None
        self._document_store: Dict[str, Dict[str, Any]] = {}
        self._persist_dir: Optional[Path] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def initialize(self, persist_dir: str) -> None:
        """
        加载 FAISS 索引和文档库（幂等，可多次调用）。

        Args:
            persist_dir: 知识库目录路径（含 ``faiss_index`` 和 ``document_store.json``）。

        Raises:
            FileNotFoundError: 若目录或 FAISS 索引文件不存在。
            ImportError: 若未安装 ``faiss-cpu`` / ``faiss-gpu``。
        """
        if self._initialized:
            return

        try:
            import faiss as _faiss
        except ImportError as exc:
            raise ImportError(
                "VectorStore 需要安装 faiss：\n"
                "  pip install faiss-cpu   # CPU 版\n"
                "  pip install faiss-gpu   # GPU 版"
            ) from exc

        persist_path = Path(persist_dir)
        index_path = persist_path / self._INDEX_FILENAME
        store_path = persist_path / self._STORE_FILENAME

        if not persist_path.exists():
            raise FileNotFoundError(
                f"知识库目录不存在：{persist_dir}\n"
                "请先运行知识库构建脚本生成索引。"
            )
        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS 索引文件不存在：{index_path}\n"
                "请先运行知识库构建脚本生成索引。"
            )

        logger.info(f"[VectorStore] 加载 FAISS 索引：{index_path}")
        self._faiss_index = _faiss.read_index(str(index_path))
        logger.info(
            f"[VectorStore] 索引加载完成  "
            f"维度={self._faiss_index.d}  向量数={self._faiss_index.ntotal}"
        )

        if store_path.exists():
            with open(store_path, "r", encoding="utf-8") as f:
                self._document_store = json.load(f)
            logger.info(
                f"[VectorStore] 文档库加载完成  共 {len(self._document_store)} 条"
            )
        else:
            logger.warning(
                f"[VectorStore] 未找到文档库文件：{store_path}，向量搜索将返回空内容"
            )

        self._persist_dir = persist_path
        self._initialized = True

    # ------------------------------------------------------------------
    # 搜索接口
    # ------------------------------------------------------------------

    def search(self, query_vector: List[float], top_k: int) -> List[DocChunk]:
        """
        向量近邻搜索，返回最相关的文档片段。

        Args:
            query_vector: 查询向量（维度须与 FAISS 索引一致）。
            top_k:        最多返回的片段数。

        Returns:
            按相似度降序排列的 :class:`~plan_a.types.DocChunk` 列表。

        Raises:
            RuntimeError: 若 ``initialize()`` 尚未调用。
        """
        if not self._initialized or self._faiss_index is None:
            raise RuntimeError(
                "VectorStore 尚未初始化，请先调用 initialize(persist_dir)"
            )

        import numpy as np

        query_array = np.array([query_vector], dtype=np.float32)
        distances, indices = self._faiss_index.search(query_array, top_k)

        doc_ids = list(self._document_store.keys())
        chunks: List[DocChunk] = []

        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(doc_ids):
                continue

            doc_id = doc_ids[int(idx)]
            doc = self._document_store.get(doc_id, {})

            # L2 距离 → 相似度分数，使用 1/(1+d) 公式保证结果在 (0, 1] 范围内
            # （L2 距离无界，直接用 1-d 会在距离 >1 时产生负数）
            score = float(1.0 / (1.0 + distances[0][i]))

            chunks.append(
                DocChunk(
                    content=doc.get("content", ""),
                    source=doc.get("source", "unknown"),
                    score=score,
                    metadata=doc.get("metadata", {}),
                )
            )

        # 1/(1+d) is always in (0, 1], so no negative-score filtering needed
        return chunks
