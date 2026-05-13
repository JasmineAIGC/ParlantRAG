"""
03_retrieval_eval.py — Parlant SDK 原生检索 + LLM 评估完整流程

使用 ``rag_module.retrieval.ParlantRetriever`` 检索，再由 ``LLMEvaluator`` 调用
既有 ``model_client.api.llm.LLMClient`` 完成综合评估；全流程不依赖 FlexRAG。

运行方式：
    python examples/03_retrieval_eval.py
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from dotenv import load_dotenv

load_dotenv()

from examples import _loader  # noqa: E402
from rag_module.evaluation import LLMEvaluator  # noqa: E402
from rag_module.retrieval import ParlantRetriever  # noqa: E402

QUERY = "what is one of the stars of  The Newcomers known for？"
TOP_K = 5


async def main() -> None:
    documents = _loader.load_hotpotqa_documents(max_examples=40)
    retriever = ParlantRetriever.from_keyword(documents)

    print(f"{'='*60}")
    print(f"🔍 检索查询：{QUERY}")
    print(f"📚 文档片段数：{len(documents)}")
    print(f"{'='*60}")

    retrieval = await retriever.retrieve(query=QUERY, top_k=TOP_K)

    if not retrieval.chunks:
        print("\n⚠️  未检索到任何相关片段，跳过评估。")
        return

    print(f"\n📄 命中 {len(retrieval.chunks)} 个片段：\n")
    for i, chunk in enumerate(retrieval.chunks, 1):
        preview = chunk.content[:100].replace("\n", " ")
        if len(chunk.content) > 100:
            preview += "..."
        print(f"  [{i}] 得分：{chunk.score:.4f}  来源：{chunk.source}")
        print(f"       {preview}\n")

    print(f"{'='*60}")
    print("📊 LLM 评估中（使用 .env 的 QWEN3_* / LLM_* 配置）...")
    print(f"{'='*60}\n")

    evaluator = LLMEvaluator()
    result = await evaluator.evaluate(question=QUERY, retrieval=retrieval)

    print(f"  verdict:    {result.verdict}")
    print(f"  relevance:  {result.relevance:.3f}")
    print(f"  coverage:   {result.coverage:.3f}")
    print(f"  confidence: {result.confidence:.3f}")
    print(f"  conclusion: {result.conclusion}")
    print(f"  hint:       {result.action_hint}")

    if result.signals:
        s = result.signals
        print("\n  量化信号：")
        print(f"    向量/检索得分  max={s.max_score:.4f}  avg={s.avg_score:.4f}  min={s.min_score:.4f}")
        print(f"    关键词覆盖率：{s.keyword_coverage:.0%}")
        if s.missing_keywords:
            print(f"    未覆盖词：{', '.join(s.missing_keywords)}")
        print(f"    来源多样性：{s.source_diversity} 个文件")


if __name__ == "__main__":
    asyncio.run(main())
