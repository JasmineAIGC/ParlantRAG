"""
RAG 主类 — 整合检索器和评估器，使用 model_client 进行模型调用。
"""
from __future__ import annotations

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()  # 加载项目根目录的 .env 文件
except ImportError:
    pass  # 如果没有安装 python-dotenv，跳过加载

from model_client.api.llm import LLMClient, LLMConfig

from parlant_rag.config import RAGConfig
from parlant_rag.retriever import Retriever, RetrievalResult, RetrievedChunk
from parlant_rag.evaluator import Evaluator, EvaluationResult
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
    基于 model_client 的 RAG 核心类。
    
    核心流程：
    1. 使用 model_client OpenAILikeEmbedding 生成查询向量
    2. 使用 llama-index 从 FAISS 知识库检索相关文档
    3. 使用 model_client LLMClient 评估检索质量
    4. 根据评估结果返回 RAGResult
    """
    
    def __init__(self, **kwargs):
        """
        初始化 RAG 实例。
        
        Args:
            **kwargs: 配置参数，会传递给 RAGConfig
                      如果未提供，会自动从环境变量（.env 文件）加载
        """
        # 如果未提供任何配置参数，从环境变量加载
        if not kwargs:
            self._config = RAGConfig.from_env()
        else:
            self._config = RAGConfig(**kwargs)
        
        self._llm_client: Optional[LLMClient] = None
        self._retriever: Optional[Retriever] = None
        self._evaluator: Optional[Evaluator] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """初始化检索器和评估器。"""
        if self._initialized:
            return
        
        logger.info("[RAG] 开始初始化  knowledge_base=%s", self._config.knowledge_base_path)
        
        # 创建检索器（连接 FAISS 知识库）
        print(f"⏳ 正在加载知识库：{self._config.knowledge_base_path}")
        self._retriever = Retriever(
            persist_dir=self._config.knowledge_base_path,
            top_k=self._config.top_k,
        )
        await self._retriever.initialize()
        
        # 创建 model_client 的 LLM 客户端（从 .env 自动加载配置）
        self._llm_client = LLMClient()
        llm_model = self._llm_client._config.model
        logger.info("[RAG] LLM 客户端初始化完成  model=%s", llm_model)
        print(f"✅ LLM 客户端初始化完成: {llm_model}")
        
        # 创建评估器
        self._evaluator = Evaluator(self._llm_client)
        
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
        logger.info("[RAG] 收到查询  question=%r", question)
        
        if not self._initialized:
            await self.initialize()
        
        # 执行检索
        retrieval = await self._retriever.retrieve(question)
        
        if retrieval.status == "error":
            logger.error("[RAG] 检索失败  error=%s", retrieval.error)
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
        
        # 根据评估结果构建回答
        answer = None
        status = evaluation.verdict
        
        if evaluation.verdict == "qualified":
            context_text = "\n\n".join([
                f"[来源: {c.source}]\n{c.content}"
                for c in retrieval.chunks
            ])
            answer = f"根据知识库内容，以下是相关信息：\n\n{context_text}"
            status = "success"
        
        logger.info("[RAG] 查询完成  status=%s  answer_length=%d", status, len(answer or ""))
        
        return RAGResult(
            query=question,
            answer=answer,
            sources=[c.source for c in retrieval.chunks],
            retrieval=retrieval,
            evaluation=evaluation,
            status=status,
        )
    
    async def add_document(self, content: str, source: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        添加文档到知识库。
        
        Args:
            content: 文档内容
            source: 文档来源
            metadata: 元数据
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info("[RAG] 添加文档  source=%s  content_length=%d", source, len(content))
        print(f"[RAG] 添加文档: {source}")
        # 实际实现需要将文档添加到向量数据库
