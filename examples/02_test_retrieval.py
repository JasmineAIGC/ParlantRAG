"""
02_test_retrieval.py — 测试文档检索功能

使用 plan_a.RAGPipeline 从知识库中检索相关文档片段。
测试问题从 knowledge_base/hotpotqa.json 中随机抽取。

运行方式：
    python examples/02_test_retrieval.py

前置条件：
    1. 已配置 .env（KNOWLEDGE_PERSIST_DIR、EMBEDDING_BASE_URL 或 EMBEDDING_MODEL_PATH）
    2. knowledge_base/hotpotqa.json 存在
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from random import Random

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from plan_a import PlanAConfig, RAGPipeline
from utils.logging import setup_logging

logger = setup_logging("test_retrieval", level="INFO", log_dir=str(_REPO_ROOT / "logs"))

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
    return Random(42).sample(questions, min(limit, len(questions)))


async def main() -> None:
    print("=" * 60)
    print("🔍  文档检索功能测试")
    print("=" * 60)
    print(f"  测试问题来源：{_HOTPOTQA_PATH}")
    logger.info("[RetrievalTest] 开始检索测试")

    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)

    print("\n⏳ 初始化 RAG 管道...")
    await pipeline.initialize()
    print("✅ RAG 管道就绪\n")

    test_queries = _load_test_queries()
    for query in test_queries:
        print(f"{'─'*60}")
        print(f"  问：{query}")
        logger.info("[RetrievalTest] 检索  query=%r", query)

        chunks = await pipeline.retrieve(query)

        if chunks:
            print(f"  ✓ 命中 {len(chunks)} 个片段")
            for i, chunk in enumerate(chunks, 1):
                print(f"\n  [{i}] 来源：{chunk.source}  相关度：{chunk.score:.4f}")
                print(f"      {chunk.content[:160].strip()}")
            logger.info("[RetrievalTest] 检索成功  chunk_count=%d", len(chunks))
        else:
            print("  ⚠️  未检索到任何相关片段")
            logger.warning("[RetrievalTest] 未命中  query=%r", query)

    print(f"\n{'='*60}")
    print("✅  检索测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
