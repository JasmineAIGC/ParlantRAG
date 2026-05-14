"""
RAG 主类 — 基于 Parlant 框架的完整检索增强生成系统。

架构设计：
  - 使用 Parlant 的接口抽象层（Embedder, SchematicGenerator, StreamingTextGenerator）
  - 使用 model_client 的实际实现（OpenAILikeEmbedding, LLMClient）
  - 通过适配器层桥接两者

核心流程（假设知识库已预生成）：
  1. 查询向量化 → 2. 向量检索 → 3. 结果评估 → 4. 回答生成

Parlant 框架特性利用：
  - NLPService: 统一的 NLP 服务接口
  - Embedder: 文本嵌入接口
  - SchematicGenerator: 结构化输出生成器
  - StreamingTextGenerator: 流式文本生成器
  - VectorDatabase: 向量数据库接口（可选）
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    SchematicGenerator,
    StreamingTextGenerator,
    StreamingTextGenerationResult,
)
from parlant.core.persistence.vector_database import VectorDatabase
from parlant.core.nlp.service import NLPService

from parlant_rag.config import RAGConfig
from parlant_rag.retriever import Retriever, RetrievalResult, RetrievedChunk
from parlant_rag.evaluator import Evaluator, EvaluationResult
from parlant_rag.adapters import (
    ModelClientEmbedder,
    ModelClientStreamingTextGenerator,
    ModelClientSchematicGenerator,
    ModelClientNLPService,
)
from utils.logging import get_logger


logger = get_logger("parlant_rag")


@dataclass
class RAGResult:
    """RAG 最终结果。"""
    query: str
    answer: Optional[str]
    sources: List[str]
    retrieval: RetrievalResult
    evaluation: EvaluationResult
    status: str  # success, no_coverage, insufficient, error


class RAG:
    """
    基于 Parlant 框架的 RAG 核心类。
    
    假设知识库已预先生成，直接从检索开始。
    
    完整的 RAG 流程：
    1. 查询向量化（使用 Parlant Embedder）
    2. 向量检索（支持 FAISS / Parlant VectorDatabase）
    3. 检索结果评估（使用 Parlant SchematicGenerator）
    4. 回答生成（使用 Parlant StreamingTextGenerator）
    
    支持两种模式：
    - 纯 model_client 模式：使用本地 FAISS + model_client LLM
    - Parlant 模式：使用 Parlant VectorDatabase + NLPService
    """
    
    def __init__(
        self,
        vector_db: Optional[VectorDatabase] = None,
        nlp_service: Optional[NLPService] = None,
        config: Optional[RAGConfig] = None,
    ):
        """
        初始化 RAG 实例。
        
        Args:
            vector_db: Parlant VectorDatabase 实例（可选）
            nlp_service: Parlant NLPService 实例（可选）
            config: RAGConfig 配置对象（可选）
        """
        self._vector_db = vector_db
        self._nlp_service = nlp_service or ModelClientNLPService()
        self._config = config or RAGConfig.from_env()
        
        self._embedder: Optional[Embedder] = None
        self._evaluator: Optional[Evaluator] = None
        self._retriever: Optional[Retriever] = None
        self._generator: Optional[StreamingTextGenerator] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化 RAG 组件。"""
        if self._initialized:
            return
        
        logger.info("[RAG] 开始初始化")
        
        # 获取 Embedder（使用 Parlant 接口，实际由 model_client 实现）
        self._embedder = await self._nlp_service.get_embedder()
        logger.info(f"[RAG] Embedder 初始化完成: {self._embedder.id}")
        
        # 创建检索器（支持 FAISS 和 Parlant VectorDatabase，假设知识库已预生成）
        self._retriever = Retriever(
            vector_db=self._vector_db,
            embedder=self._embedder,
            persist_dir=self._config.knowledge_base_path,
            collection_name=self._config.vector_db_collection,
            top_k=self._config.top_k,
        )
        await self._retriever.initialize()
        
        # 创建评估器（使用 Parlant SchematicGenerator 接口）
        eval_generator = await self._nlp_service.get_schematic_generator(dict)
        self._evaluator = Evaluator(eval_generator)
        logger.info(f"[RAG] Evaluator 初始化完成")
        
        # 获取生成器（使用 Parlant StreamingTextGenerator 接口）
        self._generator = await self._nlp_service.get_streaming_text_generator()
        logger.info(f"[RAG] Generator 初始化完成: {self._generator.id}")
        
        self._initialized = True
        logger.info("[RAG] 初始化完成")
    
    async def query(self, question: str) -> RAGResult:
        """
        执行 RAG 查询。
        
        Args:
            question: 用户问题
        
        Returns:
            RAGResult: RAG 结果
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"[RAG] 收到查询: {question!r}")
        
        # 执行检索（假设知识库已预生成）
        retrieval = await self._retriever.retrieve(question)
        
        if retrieval.status == "error":
            return RAGResult(
                query=question,
                answer=None,
                sources=[],
                retrieval=retrieval,
                evaluation=EvaluationResult(
                    verdict="no_coverage",
                    relevance=0.0,
                    coverage=0.0,
                    confidence=0.0,
                    conclusion="检索出错",
                    action_hint="检查知识库或网络连接",
                ),
                status="error",
            )
        
        # 执行评估
        evaluation = await self._evaluator.evaluate(question, retrieval)
        
        # 根据评估结果生成回答
        answer = None
        status = evaluation.verdict
        
        if evaluation.verdict == "qualified":
            context_text = "\n\n".join([
                f"[来源: {c.source}]\n{c.content}"
                for c in retrieval.chunks
            ])
            
            prompt = f"""
基于以下上下文信息回答用户问题：

上下文：
{context_text}

用户问题：{question}

请基于上下文信息给出准确、简洁的回答。
"""
            
            try:
                result: StreamingTextGenerationResult = await self._generator.generate(prompt)
                chunks = []
                async for chunk in result.stream:
                    if chunk is not None:
                        chunks.append(chunk)
                answer = "".join(chunks)
                status = "success"
                logger.info(f"[RAG] 回答生成完成")
            except Exception as e:
                logger.exception("[RAG] 回答生成失败")
                answer = f"根据知识库内容，以下是相关信息：\n\n{context_text}"
        
        return RAGResult(
            query=question,
            answer=answer,
            sources=[c.source for c in retrieval.chunks],
            retrieval=retrieval,
            evaluation=evaluation,
            status=status,
        )
