"""
basic_usage.py — 最简接入示例

演示如何用最少的代码将 RAG 模块接入 Parlant Agent。
使用 KeywordRetriever（无需向量服务）+ LLMEvaluator（LLM 配置从 .env 加载）。

运行方式（从项目根目录执行）：
  # 1. 复制并填写 .env.example → .env
  # 2. 运行：
  python -m examples.basic_usage
  # 或
  PYTHONPATH=. python examples/basic_usage.py

前置条件：
  1. pip install parlant litellm python-dotenv
  2. 启动 Parlant 服务：parlant run
  3. 配置 .env 文件（参考 .env.example）
"""
from __future__ import annotations

import asyncio

import parlant.sdk as p

from rag_module import RAGModule, LLMEvaluator

# 知识库文档（实际使用时替换为你的数据）
MY_DOCS = [
    {
        "id": "1",
        "content": "Parlant 是一个开源 AI Agent 框架，专注于确定性行为控制。",
        "source": "intro.md",
    },
    {
        "id": "2",
        "content": "Guideline 由 condition（触发条件）和 action（执行动作）两部分组成。",
        "source": "guide.md",
    },
    {
        "id": "3",
        "content": "RAGModule 支持插拔式检索器和评估器，零侵入接入任意 Parlant Agent。",
        "source": "rag.md",
    },
]


async def main() -> None:
    # LLMConfig 自动从 .env 加载（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）
    evaluator = LLMEvaluator()

    async with p.Client() as client:
        agent = await client.agents.create(name="知识助手")

        rag = RAGModule(
            documents=MY_DOCS,
            evaluator=evaluator,
        )
        await rag.register(agent)

        print(f"✅ RAG 模块已注册到 Agent: {agent.id}")
        print("   现在可以通过 Parlant 客户端向该 Agent 提问了。")


if __name__ == "__main__":
    asyncio.run(main())
