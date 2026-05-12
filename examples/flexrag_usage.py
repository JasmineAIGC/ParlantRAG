"""
flexrag_usage.py — FlexRAG 知识库 + 完整 Parlant 集成示例

演示如何：
  1. 从 .env 自动加载配置，加载 FlexRAG 构建的 FAISS 知识库
  2. 使用 LLMEvaluator 进行多信号综合评估
  3. 注册到 Parlant Agent

运行前：
  1. 使用 FlexRAG 构建知识库：
       python scripts/build_knowledge_base.py \\
           --input-dir ./my_docs \\
           --output-dir ./my_knowledge_base
  2. 复制并填写 .env.example → .env（参考 temp/llm/.env 命名）
  3. 启动 Embedding 服务（如 vLLM）
  4. 启动 Parlant 服务：parlant run
  5. pip install parlant litellm python-dotenv llama-index llama-index-vector-stores-faiss faiss-cpu httpx
"""
from __future__ import annotations

import asyncio

import parlant.sdk as p

from rag_module import RAGModule, LLMEvaluator
from rag_module.retrieval import FlexRAGRetriever


async def main() -> None:
    # ── 1. 加载 FlexRAG 知识库（所有配置从 .env 读取）──────────────────────
    retriever = FlexRAGRetriever.from_env()

    # ── 2. 初始化 LLM 评估器（所有配置从 .env 读取）────────────────────────
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
