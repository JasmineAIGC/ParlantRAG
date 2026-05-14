"""
04_rag_qa_demo.py — 独立 RAG 问答完整流程演示

演示不依赖 Parlant Server 的完整 RAG 问答流程：
  检索（plan_a.RAGPipeline）→ LLM 回答（model_client.LLMClient）

测试问题从 knowledge_base/hotpotqa.json 中随机抽取。

运行方式：
    python examples/04_rag_qa_demo.py

前置条件：
    1. 已配置 .env（KNOWLEDGE_PERSIST_DIR、EMBEDDING_*、QWEN3_* 或 LLM_*）
    2. knowledge_base/hotpotqa.json 存在

如需体验 Parlant Agent 集成，请参阅 examples/05_plan_a_agent.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from random import Random
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from plan_a import PlanAConfig, RAGPipeline
from plan_a.core.types import DocChunk
from model_client.api.llm import LLMClient
from utils.logging import setup_logging

logger = setup_logging("rag_qa_demo", level="INFO", log_dir=str(_REPO_ROOT / "logs"))

_HOTPOTQA_PATH = _REPO_ROOT / "knowledge_base" / "hotpotqa.json"


def _load_test_queries(limit: int = 3) -> list[str]:
    with _HOTPOTQA_PATH.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    questions = [
        item["question"].strip()
        for item in samples
        if isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"].strip()
    ]
    if not questions:
        raise ValueError(f"未从数据集读取到可用问题：{_HOTPOTQA_PATH}")
    return Random(0).sample(questions, min(limit, len(questions)))


async def _answer(llm: LLMClient, question: str, chunks: List[DocChunk]) -> str:
    """根据检索到的文档片段使用 LLM 生成回答。"""
    if not chunks:
        return "知识库中未找到与该问题相关的内容，无法回答。"

    context_text = "\n\n".join(
        f"[来源: {c.source}]\n{c.content}"
        for c in chunks
    )
    prompt = f"""请根据以下知识库内容简洁回答用户问题。若内容不足以回答，请如实说明。

用户问题：{question}

知识库内容：
{context_text}

回答："""
    try:
        return await llm.chat_async([{"role": "user", "content": prompt}])
    except Exception as exc:
        logger.error("[RAGDemo] LLM 调用失败: %s", exc)
        return f"LLM 调用失败，无法生成回答：{exc}"


async def main() -> None:
    print("=" * 60)
    print("🤖  独立 RAG 问答演示（无需 Parlant Server）")
    print("    检索：plan_a.RAGPipeline  |  回答：model_client.LLMClient")
    print("=" * 60)
    print(f"  测试问题来源：{_HOTPOTQA_PATH}")
    logger.info("[RAGDemo] 开始 RAG 问答演示")

    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()

    print("\n⏳ 初始化 RAG 管道...")
    await pipeline.initialize()
    print("✅ RAG 管道就绪\n")

    test_queries = _load_test_queries()
    for query in test_queries:
        print(f"{'─'*60}")
        print(f"  问：{query}")
        logger.info("[RAGDemo] query=%r", query)

        # 1. 检索
        chunks = await pipeline.retrieve(query)
        if chunks:
            top = chunks[0]
            print(f"  ✓ 命中 {len(chunks)} 片段  最相关: {top.source} ({top.score:.3f})")
        else:
            print("  ⚠️  未命中相关内容")

        # 2. LLM 生成回答
        print("  ⏳ LLM 生成回答中...")
        answer = await _answer(llm, query, chunks)

        print(f"\n  💬 回答：")
        print(f"     {answer.strip()[:300]}")
        logger.info("[RAGDemo] 问答完成  chunk_count=%d", len(chunks))

    print(f"\n{'='*60}")
    print("✅  演示结束")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
