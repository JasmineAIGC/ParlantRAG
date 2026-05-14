"""
检索器模块 — 基于 Parlant Embedder 和向量数据库的文档检索。

核心特性：
- 支持本地 FAISS 索引和 Parlant VectorDatabase 两种模式
- 使用 Parlant Embedder 接口进行文本向量化
- 假设知识库已预先生成，直接从检索开始

检索流程：
1. 查询向量化 → 2. 向量检索 → 3. 返回结果
"""
from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.persistence.vector_database import VectorDatabase

from utils.logging import get_logger


logger = get_logger("parlant_rag.retriever")


@dataclass
class RetrievedChunk:
    """检索到的文档片段。"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索结果。"""
    query: str
    chunks: List[RetrievedChunk]
    status: str  # success, no_coverage, error
    total_count: int = 0


class Retriever:
    """
    基于 Parlant 框架的文档检索器。
    
    支持两种模式：
    1. FAISS 模式：使用本地 FAISS 索引（默认）
    2. Parlant VectorDatabase 模式：使用 Parlant 向量数据库
    
    假设知识库已预先生成，直接从检索开始。
    """
    
    def __init__(
        self,
        embedder: Embedder,
        vector_db: Optional[VectorDatabase] = None,
        persist_dir: Optional[str] = None,
        collection_name: str = "parlant_rag",
        top_k: int = 5,
    ):
        """
        初始化检索器。
        
        Args:
            embedder: Parlant Embedder 实例
            vector_db: Parlant VectorDatabase 实例（可选）
            persist_dir: FAISS 索引持久化目录（可选）
            collection_name: 向量数据库集合名称
            top_k: 检索返回的文档数量
        """
        self._embedder = embedder
        self._vector_db = vector_db
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._top_k = top_k
        
        self._faiss_index = None
        self._document_store: Dict[str, Dict[str, Any]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化检索器。"""
        if self._initialized:
            return
        
        logger.info("[Retriever] 开始初始化")
        
        if self._vector_db:
            # Parlant VectorDatabase 模式
            await self._initialize_parlant_vector_db()
        elif self._persist_dir:
            # FAISS 模式
            await self._initialize_faiss()
        else:
            raise ValueError("必须提供 vector_db 或 persist_dir 参数之一")
        
        self._initialized = True
        logger.info("[Retriever] 初始化完成")
    
    async def _initialize_parlant_vector_db(self) -> None:
        """初始化 Parlant VectorDatabase。"""
        logger.info(f"[Retriever] 使用 Parlant VectorDatabase: {self._collection_name}")
        # 确保集合存在
        await self._vector_db.ensure_collection(
            collection_name=self._collection_name,
            embedder_type=type(self._embedder),
        )
    
    async def _initialize_faiss(self) -> None:
        """初始化本地 FAISS 索引。"""
        import faiss
        
        persist_path = Path(self._persist_dir)
        index_path = persist_path / "faiss_index"
        store_path = persist_path / "document_store.json"
        
        logger.info(f"[Retriever] 使用本地 FAISS 索引: {persist_path}")
        
        if persist_path.exists() and index_path.exists():
            # 加载已存在的索引（假设知识库已预先生成）
            self._faiss_index = faiss.read_index(str(index_path))
            if store_path.exists():
                with open(store_path, "r", encoding="utf-8") as f:
                    self._document_store = json.load(f)
            logger.info(f"[Retriever] 加载已有索引，共 {len(self._document_store)} 个文档")
        else:
            # 如果索引不存在，创建空索引
            persist_path.mkdir(parents=True, exist_ok=True)
            dimensions = self._embedder.dimensions
            self._faiss_index = faiss.IndexFlatL2(dimensions)
            logger.warning(f"[Retriever] 未找到预生成的索引，创建新的空索引，维度: {dimensions}")
    
    async def retrieve(self, query: str) -> RetrievalResult:
        """
        根据查询检索相关文档。
        
        Args:
            query: 查询文本
        
        Returns:
            RetrievalResult: 检索结果
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"[Retriever] 检索查询: {query!r}")
        
        try:
            # 向量化查询
            query_embedding: EmbeddingResult = await self._embedder.embed([query])
            query_vector = query_embedding.vectors[0]
            
            if self._vector_db:
                # Parlant VectorDatabase 模式
                results = await self._vector_db.search(
                    collection_name=self._collection_name,
                    query_vector=query_vector,
                    top_k=self._top_k,
                )
                
                chunks = [
                    RetrievedChunk(
                        content=result.text,
                        source=result.metadata.get("source", "unknown"),
                        score=1 - result.distance,
                        metadata=result.metadata,
                    )
                    for result in results
                ]
            else:
                # FAISS 模式
                import numpy as np
                
                query_array = np.array([query_vector], dtype=np.float32)
                distances, indices = self._faiss_index.search(query_array, self._top_k)
                
                chunks = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1:
                        doc_id = list(self._document_store.keys())[idx]
                        doc = self._document_store[doc_id]
                        chunks.append(RetrievedChunk(
                            content=doc["content"],
                            source=doc["source"],
                            score=1 - distances[0][i],
                            metadata=doc["metadata"],
                        ))
            
            # 过滤低相似度结果
            chunks = [c for c in chunks if c.score >= 0.1]
            
            status = "success" if chunks else "no_coverage"
            
            logger.info(f"[Retriever] 检索完成，找到 {len(chunks)} 个相关文档")
            
            return RetrievalResult(
                query=query,
                chunks=chunks,
                status=status,
                total_count=len(chunks),
            )
        
        except Exception as e:
            logger.exception("[Retriever] 检索失败")
            return RetrievalResult(
                query=query,
                chunks=[],
                status="error",
                total_count=0,
            )
