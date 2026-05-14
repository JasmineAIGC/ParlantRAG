"""
03_test_retrieval_eval.py — 测试检索评估功能

验证评估器能否正确评估检索结果的质量。

运行方式：
    python examples/03_test_retrieval_eval.py
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
logger = setup_logging("test_retrieval_eval", level="INFO", log_dir="../logs")


async def main() -> None:
    # 测试查询
    test_query = "what is one of the stars of The Newcomers known for?"
    
    print(f"{'='*60}")
    print(f"🔍 检索查询：{test_query}")
    print(f"{'='*60}")
    logger.info(f"[EvalTest] 开始检索评估测试  query={test_query!r}")

    # 创建 RAG 实例
    rag = RAG()
    
    # 执行检索和评估
    result = await rag.query(test_query)

    # 输出评估结果
    print("\n📊 评估结果：")
    print(f"  评估结论: {result.evaluation.verdict}")
    print(f"  相关性: {result.evaluation.relevance:.2f}")
    print(f"  覆盖度: {result.evaluation.coverage:.2f}")
    print(f"  置信度: {result.evaluation.confidence:.2f}")
    print(f"  说明: {result.evaluation.conclusion}")
    print(f"  建议: {result.evaluation.action_hint}")
    
    logger.info(
        "[EvalTest] 评估完成  verdict=%s  relevance=%.2f  coverage=%.2f  confidence=%.2f",
        result.evaluation.verdict,
        result.evaluation.relevance,
        result.evaluation.coverage,
        result.evaluation.confidence
    )


if __name__ == "__main__":
    asyncio.run(main())
