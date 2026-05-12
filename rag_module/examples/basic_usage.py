"""
最简使用示例：使用内置 KeywordRetriever 和 RuleBasedEvaluator，3 行接入。

运行前请先启动 Parlant 服务：
    parlant run
"""
import asyncio
import parlant.sdk as p

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rag_module import RAGModule

MY_DOCS = [
    {"id": "1", "content": "Parlant 是一个开源 AI Agent 框架，专注于确定性行为控制。", "source": "intro.md"},
    {"id": "2", "content": "Guideline 由 condition（条件）和 action（动作）两部分组成。", "source": "guide.md"},
    {"id": "3", "content": "RAGModule 支持插拔式检索器和评价器，零侵入接入任意 Parlant Agent。", "source": "rag.md"},
]


async def main() -> None:
    async with p.Client() as client:
        agent = await client.agents.create(
            name="知识助手",
        )

        # ── 接入 RAG 模块，就这 3 行 ──────────────────────────
        rag = RAGModule(documents=MY_DOCS)
        await rag.register(agent)
        # ─────────────────────────────────────────────────────

        print(f"✅ RAG 模块已注册到 Agent: {agent.id}")


if __name__ == "__main__":
    asyncio.run(main())
