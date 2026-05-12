"""
standalone_test.py — 独立测试（不需要启动 Parlant 服务）

直接调用检索器和评估器，验证检索与评估流程是否正常工作。
可在配置好 LLM API Key 后直接运行。

pip install openai
"""
from __future__ import annotations

import asyncio
import os

from rag_module.retrieval import KeywordRetriever
from rag_module.evaluation import LLMEvaluator

DOCS = [
    {
        "id": "1",
        "content": "Parlant 是一个开源 AI Agent 框架，专注于确定性行为控制。",
        "source": "intro.md",
    },
    {
        "id": "2",
        "content": "Guideline 由 condition（触发条件）和 action（执行动作）两部分组成，用于精确控制 Agent 行为。",
        "source": "guide.md",
    },
    {
        "id": "3",
        "content": "RAGModule 支持插拔式检索器和评估器，零侵入接入任意 Parlant Agent。",
        "source": "rag.md",
    },
    {
        "id": "4",
        "content": "向量检索基于余弦相似度（或内积）计算文档与查询的语义相关性。",
        "source": "vector_search.md",
    },
]


async def main() -> None:
    # ── 检索 ──────────────────────────────────────────────────────────────
    retriever = KeywordRetriever(DOCS)

    query = "Parlant 的 Guideline 是什么"
    print(f"\n{'='*60}")
    print(f"🔍 检索查询：{query}")
    print(f"{'='*60}")

    retrieval = await retriever.retrieve(query, top_k=3)

    print(f"\n📄 检索到 {len(retrieval.chunks)} 个片段：")
    for chunk in retrieval.chunks:
        print(f"  [{chunk.score:.4f}] {chunk.source}")
        print(f"           {chunk.content[:80]}...")

    # ── 评估 ──────────────────────────────────────────────────────────────
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        print(
            "\n⚠️  未设置 LLM_API_KEY 环境变量，跳过 LLM 评估。\n"
            "    设置方法：export LLM_API_KEY=sk-your-key"
        )
        return

    evaluator = LLMEvaluator(
        api_key=api_key,
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.0,
    )

    print(f"\n{'='*60}")
    print("📊 LLM 多信号评估中...")
    print(f"{'='*60}")

    result = await evaluator.evaluate(question=query, retrieval=retrieval)

    print(f"\n  verdict:    {result.verdict}")
    print(f"  relevance:  {result.relevance:.3f}")
    print(f"  coverage:   {result.coverage:.3f}")
    print(f"  confidence: {result.confidence:.3f}")
    print(f"  conclusion: {result.conclusion}")
    print(f"  hint:       {result.action_hint}")

    if result.signals:
        s = result.signals
        print(f"\n  量化信号：")
        print(f"    向量得分  max={s.max_score:.4f}  avg={s.avg_score:.4f}  min={s.min_score:.4f}")
        print(f"    关键词覆盖率：{s.keyword_coverage:.0%}")
        if s.missing_keywords:
            print(f"    未覆盖词：{', '.join(s.missing_keywords)}")
        print(f"    来源多样性：{s.source_diversity} 个文件")


if __name__ == "__main__":
    asyncio.run(main())
