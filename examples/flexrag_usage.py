"""
flexrag_usage.py — FlexRAG 知识库 + 完整 Parlant 集成示例

演示如何：
  1. 加载 FlexRAG scripts/build_knowledge_base.py 构建的 FAISS 知识库
     （使用与 FlexRAG MultiVectorRetriever / OpenAILikeEmbedding 兼容的调用方式）
  2. 使用 LLMEvaluator 进行多信号综合评估（LLM 通过 litellm 调用）
  3. 注册到 Parlant Agent

运行前：
  1. 使用 FlexRAG 构建知识库：
       python scripts/build_knowledge_base.py \\
           --input-dir ./my_docs \\
           --output-dir ./my_knowledge_base
  2. 启动 Embedding 服务（如 vLLM）
  3. 配置 .env 文件（参考 .env.example）
  4. 启动 Parlant 服务：parlant run
  5. pip install parlant litellm python-dotenv llama-index llama-index-vector-stores-faiss faiss-cpu httpx
"""
from __future__ import annotations

import asyncio
import os

import parlant.sdk as p

from rag_module import RAGModule, LLMEvaluator, LLMConfig
from rag_module.retrieval import FlexRAGRetriever


async def main() -> None:
    # ── 1. 初始化检索器（加载 FlexRAG 构建的知识库）──────────────────────
    #    使用 OpenAILikeEmbedding（与 FlexRAG 构建阶段完全一致的 Embedding 接口）
    retriever = FlexRAGRetriever(
        persist_dir=os.environ.get("KNOWLEDGE_BASE_DIR", "./my_knowledge_base"),
        embedding_base_url=os.environ.get("EMBEDDING_BASE_URL", "http://127.0.0.1:8018/v1"),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY", "sk-xxxx"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
        top_k=5,
        bm25_weight=0.3,   # 30% BM25 + 70% 向量混合（需要 bm25_index/ 子目录）
    )

    # ── 2. 初始化 LLM 多信号评估器──────────────────────────────────────────
    #    LLMConfig 从 .env 自动加载（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）
    #    也可手动指定：LLMConfig(api_key="sk-...", model="Qwen2.5-7B-Instruct", base_url="...")
    evaluator = LLMEvaluator()

    # ── 3. 注册到 Parlant Agent ───────────────────────────────────────────
    async with p.Client() as client:
        agent = await client.agents.create(name="FlexRAG知识助手")

        rag = RAGModule(retriever=retriever, evaluator=evaluator)
        await rag.register(agent)

        print(f"✅ FlexRAG RAG 模块已注册到 Agent: {agent.id}")
        print("   现在可以通过 Parlant 客户端向该 Agent 提问了。")


if __name__ == "__main__":
    asyncio.run(main())
