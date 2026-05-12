"""
03_retrieval_eval.py — 检索 + LLM 评估完整流程

加载 FlexRAG 构建的 FAISS 知识库，执行向量检索，
再由 LLMEvaluator 对检索结果做多信号综合评估（相关性、覆盖度、置信度）。

运行方式（从项目根目录执行）：
    PYTHONPATH=. python examples/03_retrieval_eval.py

前置条件：
    1. 用 FlexRAG scripts/build_knowledge_base.py 构建好知识库
    2. 复制并填写 .env.example → .env，配置以下变量（参考 temp/llm/.env）：
         KNOWLEDGE_PERSIST_DIR=./knowledge_base
         EMBEDDING_BASE_URL=http://10.x.x.x:8018/v1
         EMBEDDING_MODEL=bge-large-zh-v1.5
         QWEN3_API_KEY=sk-xxxx
         QWEN3_BASE_URL=http://10.x.x.x:18023/v1
         QWEN3_MODEL=Qwen2.5-7B-Instruct
    3. 启动 Embedding 服务和 LLM 服务
    4. pip install litellm python-dotenv llama-index llama-index-vector-stores-faiss faiss-cpu httpx
"""
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()

from rag_module.retrieval import FlexRAGRetriever  # noqa: E402
from rag_module.evaluation import LLMEvaluator     # noqa: E402

# 修改为你的实际查询词
QUERY = "请问知识库中有哪些相关内容？"
TOP_K = 5


async def main() -> None:
    # ── 1. 加载知识库 ──────────────────────────────────────────────────────
    print("⏳ 正在加载知识库...")
    retriever = FlexRAGRetriever.from_env()
    print("✅ 知识库加载完成\n")

    # ── 2. 检索 ────────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print(f"🔍 检索查询：{QUERY}")
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

    # ── 3. LLM 评估 ────────────────────────────────────────────────────────
    print(f"{'='*60}")
    print("📊 LLM 评估中（参考 temp/llm/.env 的 QWEN3_* 配置）...")
    print(f"{'='*60}\n")

    # LLMEvaluator 从 .env 自动加载 LLM 配置（QWEN3_* 优先，参考 temp/llm/.env）
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
        print(f"\n  量化信号：")
        print(f"    向量得分  max={s.max_score:.4f}  avg={s.avg_score:.4f}  min={s.min_score:.4f}")
        print(f"    关键词覆盖率：{s.keyword_coverage:.0%}")
        if s.missing_keywords:
            print(f"    未覆盖词：{', '.join(s.missing_keywords)}")
        print(f"    来源多样性：{s.source_diversity} 个文件")


if __name__ == "__main__":
    asyncio.run(main())
