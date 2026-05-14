"""
parlant_rag — 基于 model_client 的 RAG 模块。

核心功能：
  - 文档检索：使用 model_client OpenAILikeEmbedding + FAISS
  - 检索评估：使用 model_client LLMClient
  - RAG 问答：整合检索+评估+回答

文件结构：
  - config.py     : 配置管理
  - retriever.py  : 检索器
  - evaluator.py  : 评估器
  - rag.py        : RAG 主类

日志模块位于 utils/logging.py

使用方式：
    from parlant_rag import RAG
    
    # 创建 RAG 实例（自动从 .env 加载配置）
    rag = RAG()
    
    # 执行 RAG 查询
    result = await rag.query("你的问题")
    
    # 获取结果
    print(result.answer)
    print(result.sources)
"""
from parlant_rag.config import RAGConfig
from parlant_rag.retriever import Retriever, RetrievalResult, RetrievedChunk
from parlant_rag.evaluator import Evaluator, EvaluationResult
from parlant_rag.rag import RAG, RAGResult
from utils.logging import setup_logging, get_logger

__all__ = [
    "RAGConfig",
    "Retriever",
    "RetrievalResult",
    "RetrievedChunk",
    "Evaluator",
    "EvaluationResult",
    "RAG",
    "RAGResult",
    "setup_logging",
    "get_logger",
]
