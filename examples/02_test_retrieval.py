"""
02_test_retrieval.py — 测试文档检索功能

验证检索器能否从知识库中找到相关文档片段。

运行方式：
    python examples/02_test_retrieval.py
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from parlant_rag import RAG
from utils.logging import setup_logging

# 设置日志（输出到终端和文件）
logger = setup_logging("test_retrieval", level="INFO", log_dir="../logs")


async def main() -> None:
    # 测试查询
    test_query = "what is one of the stars of The Newcomers known for?"
    
    print(f"{'='*60}")
    print(f"🔍 检索查询：{test_query}")
    print(f"{'='*60}")
    logger.info(f"[RetrievalTest] 开始检索测试  query={test_query!r}")

    # 创建 RAG 实例
    rag = RAG()
    
    # 执行检索
    result = await rag.query(test_query)

    # 输出检索结果
    print("\n📋 检索结果：")
    if result.retrieval.chunks:
        for i, chunk in enumerate(result.retrieval.chunks, 1):
            print(f"\n  [{i}] 来源: {chunk.source}")
            print(f"     相关性: {chunk.score}")
            print(f"     内容:\n{chunk.content}")
        logger.info(f"[RetrievalTest] 检索成功  chunk_count={len(result.retrieval.chunks)}")
    else:
        print("  ⚠️  未检索到任何相关片段。")
        logger.warning("[RetrievalTest] 未检索到任何相关片段")


if __name__ == "__main__":
    asyncio.run(main())
