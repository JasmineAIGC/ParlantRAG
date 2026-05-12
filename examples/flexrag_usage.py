"""
flexrag_usage.py — FlexRAG 知识库 + 完整 Parlant 集成示例

演示如何：
  1. 加载 FlexRAG scripts/build_knowledge_base.py 构建的 FAISS 知识库
  2. 使用 LLMEvaluator 进行多信号综合评估
  3. 注册到 Parlant Agent

运行前：
  1. 使用 FlexRAG 构建知识库：
       python scripts/build_knowledge_base.py \
           --input-dir ./my_docs \
           --output-dir ./my_knowledge_base \
           --embedding-base-url http://127.0.0.1:8018/v1 \
           --embedding-model Qwen3-Embedding-0.6B
  2. 启动 Embedding 服务（如 vLLM）
  3. 启动 Parlant 服务：parlant run
  4. 设置环境变量 LLM_API_KEY
  5. pip install parlant openai llama-index llama-index-vector-stores-faiss faiss-cpu httpx
"""
from __future__ import annotations

import asyncio
import os

import parlant.sdk as p

from rag_module import RAGModule, LLMEvaluator
from rag_module.retrieval import FlexRAGRetriever


async def main() -> None:
    # ── 1. 初始化检索器（加载 FlexRAG 构建的知识库） ─────────────────────
    retriever = FlexRAGRetriever(
        persist_dir=os.environ.get("KNOWLEDGE_BASE_DIR", "./my_knowledge_base"),
        embedding_base_url=os.environ.get("EMBEDDING_URL", "http://127.0.0.1:8018/v1"),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY", "sk-xxxx"),
        embedding_model=os.environ.get("EMBEDDING_MODEL", "Qwen3-Embedding-0.6B"),
        top_k=5,
        bm25_weight=0.3,   # 30% BM25 + 70% 向量混合（需要 bm25_index/ 子目录）
    )

    # ── 2. 初始化 LLM 多信号评估器 ───────────────────────────────────────
    evaluator = LLMEvaluator(
        api_key=os.environ.get("LLM_API_KEY", "sk-YOUR-KEY-HERE"),
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        # base_url="https://your-openai-compatible-endpoint/v1",  # 可选
        temperature=0.0,
    )

    # ── 3. 注册到 Parlant Agent ───────────────────────────────────────────
    async with p.Client() as client:
        agent = await client.agents.create(name="FlexRAG知识助手")

        rag = RAGModule(retriever=retriever, evaluator=evaluator)
        await rag.register(agent)

        print(f"✅ FlexRAG RAG 模块已注册到 Agent: {agent.id}")
        print("   现在可以通过 Parlant 客户端向该 Agent 提问了。")


if __name__ == "__main__":
    asyncio.run(main())
