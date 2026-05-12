"""
进阶使用示例：

1. 使用 FlexRAGRetriever 加载 FlexRAG 构建的 FAISS 知识库
2. 使用 LLMEvaluator 进行语义级评价（可选）
3. 演示自定义 Retriever 和 Evaluator

运行前：
  1. 使用 FlexRAG build_knowledge_base.py 构建知识库到 ./my_knowledge_base/
  2. 启动 Embedding 服务（如 vLLM）
  3. 启动 Parlant 服务：parlant run
"""
import asyncio
import parlant.sdk as p

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rag_module import (
    RAGModule,
    BaseRetriever,
    BaseEvaluator,
    Chunk,
    RetrievalResult,
    EvaluationResult,
)
from rag_module.retriever import FlexRAGRetriever
from rag_module.evaluator import LLMEvaluator


# ── 示例一：使用 FlexRAGRetriever + RuleBasedEvaluator ─────────────────────

async def example_flexrag() -> None:
    """加载 FlexRAG 构建的知识库，使用规则评价器。"""
    retriever = FlexRAGRetriever(
        persist_dir="./my_knowledge_base",       # FlexRAG 输出目录
        embedding_base_url="http://127.0.0.1:8018/v1",
        embedding_api_key="sk-xxxx",
        embedding_model="Qwen3-Embedding-0.6B",
        top_k=5,
        bm25_weight=0.3,   # 30% BM25 + 70% 向量检索（需要 bm25_index/ 目录存在）
    )

    async with p.Client() as client:
        agent = await client.agents.create(name="FlexRAG知识助手")
        rag = RAGModule(retriever=retriever)
        await rag.register(agent)
        print(f"✅ FlexRAG RAG 模块已注册到 Agent: {agent.id}")


# ── 示例二：使用 FlexRAGRetriever + LLMEvaluator ───────────────────────────

async def example_llm_eval() -> None:
    """加载 FlexRAG 构建的知识库，使用 LLM 评价器。"""
    retriever = FlexRAGRetriever(
        persist_dir="./my_knowledge_base",
        embedding_base_url="http://127.0.0.1:8018/v1",
        embedding_api_key="sk-xxxx",
        embedding_model="Qwen3-Embedding-0.6B",
    )
    evaluator = LLMEvaluator(
        api_key="sk-...",
        model="gpt-4o-mini",
        # base_url="https://your-openai-compatible-endpoint/v1",
    )

    async with p.Client() as client:
        agent = await client.agents.create(name="严格知识助手")
        rag = RAGModule(retriever=retriever, evaluator=evaluator)
        await rag.register(agent)
        print(f"✅ LLM 评价 RAG 模块已注册到 Agent: {agent.id}")


# ── 示例三：完全自定义检索器和评价器 ──────────────────────────────────────

class StrictEvaluator(BaseEvaluator):
    """自定义评价器：只有最高得分片段 >= 0.7 才判为合格。"""

    async def evaluate(self, question: str, retrieval: RetrievalResult) -> EvaluationResult:
        if not retrieval.chunks:
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0, coverage=0.0, confidence=0.0,
                conclusion="未检索到内容。",
                action_hint="告知用户超出知识库范围。",
            )

        best_score = max(c.score for c in retrieval.chunks)
        avg_score = sum(c.score for c in retrieval.chunks) / len(retrieval.chunks)

        if best_score >= 0.7:
            return EvaluationResult(
                verdict="qualified",
                relevance=avg_score, coverage=best_score, confidence=best_score,
                conclusion=f"找到高质量匹配片段（得分 {best_score:.2f}）。",
                action_hint="可直接基于检索内容作答。",
            )
        else:
            return EvaluationResult(
                verdict="insufficient",
                relevance=avg_score, coverage=best_score, confidence=best_score,
                conclusion=f"最佳匹配得分仅 {best_score:.2f}，低于严格阈值 0.7。",
                action_hint="建议换词重新提问。",
            )


async def example_custom() -> None:
    """使用自定义评价器。"""
    from rag_module import KeywordRetriever

    docs = [
        {"id": "1", "content": "Parlant 支持多轮对话与状态管理。", "source": "parlant.md"},
        {"id": "2", "content": "向量检索基于余弦相似度计算文档相关性。", "source": "rag.md"},
    ]

    async with p.Client() as client:
        agent = await client.agents.create(name="自定义严格助手")
        rag = RAGModule(
            retriever=KeywordRetriever(docs),
            evaluator=StrictEvaluator(),
        )
        await rag.register(agent)
        print(f"✅ 自定义 RAG 模块已注册到 Agent: {agent.id}")


# ── 直接测试检索和评价（不依赖 Parlant 服务）─────────────────────────────

async def standalone_test() -> None:
    """直接调用检索器和评价器，无需启动 Parlant 服务。"""
    from rag_module import KeywordRetriever, RuleBasedEvaluator

    docs = [
        {"id": "1", "content": "Parlant 是一个开源 AI Agent 框架，专注于确定性行为控制。", "source": "intro.md"},
        {"id": "2", "content": "Guideline 由 condition 和 action 组成，用于精确控制 Agent 行为。", "source": "guide.md"},
        {"id": "3", "content": "RAGModule 支持插拔式检索器和评价器。", "source": "rag.md"},
    ]

    retriever = KeywordRetriever(docs)
    evaluator = RuleBasedEvaluator()

    query = "Parlant 的 Guideline 是什么"
    print(f"\n🔍 检索查询：{query}")

    result = await retriever.retrieve(query, top_k=3)
    print(f"\n📄 检索结果（共 {len(result.chunks)} 条）：")
    for chunk in result.chunks:
        print(f"  [{chunk.score:.4f}] {chunk.source}: {chunk.content[:60]}...")

    eval_result = await evaluator.evaluate(question=query, retrieval=result)
    print(f"\n📊 评价结果：")
    print(f"  verdict:    {eval_result.verdict}")
    print(f"  relevance:  {eval_result.relevance}")
    print(f"  coverage:   {eval_result.coverage}")
    print(f"  confidence: {eval_result.confidence}")
    print(f"  conclusion: {eval_result.conclusion}")
    print(f"  hint:       {eval_result.action_hint}")


if __name__ == "__main__":
    # 直接运行时，执行独立测试（无需 Parlant 服务）
    asyncio.run(standalone_test())
