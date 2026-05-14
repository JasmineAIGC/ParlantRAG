"""
检索器 — 使用 model_client 和 llama-index 连接 FAISS 知识库。

基于 FlexRAGRetriever 的实现，支持加载 FAISS 索引并执行向量检索。
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from model_client.api.embedding import OpenAILikeEmbedding
from model_client.local.embedding import SentenceTransformerEmbedding
from model_client._utils import first_env

from utils.logging import get_logger


logger = get_logger("parlant_rag.retriever")
_FAIISS_FILE = "faiss_index.bin"


@dataclass
class RetrievedChunk:
    """检索到的文档片段。"""
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = None


@dataclass
class RetrievalResult:
    """检索结果。"""
    query: str
    chunks: List[RetrievedChunk]
    status: str  # success, no_results, error
    error: Optional[str] = None


class Retriever:
    """基于 model_client 和 llama-index 的 FAISS 知识库检索器。"""
    
    def __init__(self, persist_dir: str, top_k: int = 5):
        self._persist_dir = persist_dir
        self._top_k = top_k
        self._index = None
        self._embed_model = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化检索器，加载 FAISS 索引。"""
        if self._initialized:
            return
        
        try:
            import faiss as _faiss
            from llama_index.core import Settings, StorageContext, load_index_from_storage
            from llama_index.vector_stores.faiss import FaissVectorStore
        except ImportError:
            raise ImportError(
                "Retriever 需要安装：\n"
                "pip install llama-index llama-index-vector-stores-faiss faiss-cpu httpx"
            )
        
        faiss_path = os.path.join(self._persist_dir, _FAIISS_FILE)
        if not os.path.exists(faiss_path):
            raise FileNotFoundError(
                f"未找到 FAISS 索引文件：{faiss_path}\n"
                "请先构建知识库或将 KNOWLEDGE_PERSIST_DIR 设置为正确目录。"
            )
        
        # --- 选择 Embedding 后端 ---
        model_path = os.getenv("EMBEDDING_MODEL_PATH")
        base_url = os.getenv("EMBEDDING_BASE_URL")
        
        if model_path:
            # 本地 sentence-transformers 模式
            model = os.getenv("EMBEDDING_MODEL") or os.path.basename(model_path.rstrip("/\\"))
            self._embed_model = SentenceTransformerEmbedding(
                model_path=model_path,
                model_name=model,
            )
            logger.info("[Retriever] 使用本地 Embedding 模型  model=%s  path=%s", model, model_path)
        elif base_url:
            # API 模式：OpenAI 兼容 Embedding 服务
            api_key = first_env(["EMBEDDING_API_KEY", "QWEN3_API_KEY", "LLM_API_KEY"])
            model = first_env(["EMBEDDING_MODEL", "QWEN3_MODEL", "LLM_MODEL"], default="text-embedding-3-small")
            self._embed_model = OpenAILikeEmbedding(
                base_url=base_url,
                model_name=model,
                api_key=api_key,
            )
            logger.info("[Retriever] 使用 API Embedding 服务  model=%s  url=%s", model, base_url)
        else:
            raise ValueError(
                "未配置 Embedding 后端。\n"
                "请设置以下其中一组环境变量：\n"
                "  本地模式：EMBEDDING_MODEL_PATH=/path/to/model\n"
                "  API  模式：EMBEDDING_BASE_URL=http://host:port/v1  EMBEDDING_MODEL=model-name"
            )
        
        Settings.embed_model = self._embed_model
        
        # --- 加载 FAISS 索引 ---
        faiss_index = _faiss.read_index(faiss_path)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=self._persist_dir,
        )
        self._index = load_index_from_storage(storage_context)
        
        doc_count = len(self._index.docstore.docs)
        logger.info("[Retriever] 知识库加载成功  doc_count=%d", doc_count)
        print(f"✅ 知识库加载完成（{doc_count} 个文档片段）")
        
        self._initialized = True
    
    async def retrieve(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """执行文档检索。"""
        try:
            if not self._initialized:
                await self.initialize()
            
            k = top_k if top_k is not None else self._top_k
            
            logger.info("[Retriever] 检索开始  query=%r  top_k=%d", query, k)
            
            # 使用 llama-index 的异步 aretrieve()
            retriever = self._index.as_retriever(similarity_top_k=k)
            nodes = await retriever.aretrieve(query)
            
            logger.info("[Retriever] 检索返回 %d 个节点", len(nodes))
            
            # 转换为 RetrievedChunk
            chunks = []
            for idx, node in enumerate(nodes):
                meta = node.metadata or {}
                source = (
                    meta.get("file_name")
                    or meta.get("title")
                    or meta.get("source")
                    or "unknown"
                )
                chunks.append(RetrievedChunk(
                    content=node.get_content(),
                    source=source,
                    score=round(node.score or 0.0, 4),
                    metadata=meta,
                ))
            
            status = "success" if chunks else "no_results"
            logger.info("[Retriever] 检索完成  status=%s  chunk_count=%d", status, len(chunks))
            
            return RetrievalResult(
                query=query,
                chunks=chunks,
                status=status,
            )
        
        except Exception as e:
            logger.exception("[Retriever] 检索失败  query=%r", query)
            return RetrievalResult(
                query=query,
                chunks=[],
                status="error",
                error=str(e),
            )
