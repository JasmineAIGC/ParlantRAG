"""
parlant_rag — 基于 Parlant 框架 + model_client 的完整 RAG 系统。

架构设计：
  - 使用 Parlant 的接口抽象层（Embedder, SchematicGenerator, StreamingTextGenerator）
  - 使用 model_client 的实际实现（OpenAILikeEmbedding, LLMClient）
  - 通过适配器层桥接两者

核心组件：
  - adapters: Parlant 接口与 model_client 的适配器
  - config: RAG 配置管理
  - retriever: 文档检索器（支持 FAISS / Parlant VectorDatabase）
  - evaluator: 检索结果评估器
  - rag: RAG 主类

Parlant 框架特性利用：
  - NLPService: 统一的 NLP 服务接口
  - Embedder: 文本嵌入接口
  - SchematicGenerator: 结构化输出生成器
  - StreamingTextGenerator: 流式文本生成器
  - VectorDatabase: 向量数据库接口（可选）

完整 RAG 流程：
  1. 文档加载与向量化（使用 Parlant Embedder）
  2. 向量检索（支持 FAISS / Parlant VectorDatabase）
  3. 检索结果评估（使用 Parlant SchematicGenerator）
  4. 回答生成（使用 Parlant StreamingTextGenerator）

使用方式：
    from parlant_rag import RAG, RAGConfig
    
    # 方式1：纯 model_client 模式（推荐）
    config = RAGConfig.from_env()
    rag = RAG(config=config)
    
    # 添加文档
    await rag.add_documents([
        {"content": "文档内容...", "source": "doc1.txt"}
    ])
    
    # 查询
    result = await rag.query("你的问题")
"""
from parlant_rag.config import RAGConfig
from parlant_rag.retriever import Retriever, RetrievalResult, RetrievedChunk
from parlant_rag.evaluator import Evaluator, EvaluationResult
from parlant_rag.rag import RAG, RAGResult
from parlant_rag.adapters import (
    ModelClientEmbedder,
    ModelClientStreamingTextGenerator,
    ModelClientSchematicGenerator,
    ModelClientNLPService,
)

__all__ = [
    "RAGConfig",
    "Retriever",
    "RetrievalResult",
    "RetrievedChunk",
    "Evaluator",
    "EvaluationResult",
    "RAG",
    "RAGResult",
    "ModelClientEmbedder",
    "ModelClientStreamingTextGenerator",
    "ModelClientSchematicGenerator",
    "ModelClientNLPService",
]
