"""
04_rag_qa_demo.py — RAG 问答完整流程演示（基于 Parlant 框架）

演示完整的 RAG 问答流程：检索 → 评估 → 回答。

运行方式：
    python examples/04_rag_qa_demo.py

前置条件：
    1. 配置 Parlant 环境变量（参考 third_party/parlant 的配置方式）
    2. 确保向量数据库已配置并包含知识库
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 添加 Parlant 框架到路径
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "parlant", "src"))

from dotenv import load_dotenv
load_dotenv()

from parlant_rag import RAG
from utils.logging import setup_logging

# 设置日志（输出到终端和文件）
logger = setup_logging("rag_qa_demo", level="INFO", log_dir="../logs")


async def main() -> None:
    # 测试查询
    test_query = "what is one of the stars of The Newcomers known for?"
    
    print(f"{'='*60}")
    print(f"🤖 RAG 问答演示（基于 Parlant 框架）")
    print(f"{'='*60}")
    print(f"查询问题：{test_query}")
    logger.info(f"[RAGDemo] 开始 RAG 问答演示  query={test_query!r}")

    try:
        # 创建 Parlant 应用
        from parlant import Application
        
        # 初始化 Parlant 应用
        print("\n⏳ 正在初始化 Parlant 应用...")
        app = Application()
        
        # 获取 Parlant 服务
        print("⏳ 正在获取向量数据库...")
        vector_db = await app.vector_database()
        print("✅ 向量数据库获取完成")
        
        print("⏳ 正在获取 NLP 服务...")
        nlp_service = await app.nlp_service()
        print("✅ NLP 服务获取完成")
        
        # 创建 RAG 实例（基于 Parlant）
        rag = RAG(
            vector_db=vector_db,
            nlp_service=nlp_service,
            collection_name="rag_knowledge_base",
            top_k=5,
        )
        
        # 执行完整 RAG 查询
        print("\n⏳ 正在执行 RAG 查询...")
        result = await rag.query(test_query)

        # 输出结果
        print(f"\n{'─'*60}")
        print("📋 检索到的文档：")
        print(f"{'─'*60}")
        if result.retrieval.chunks:
            for i, chunk in enumerate(result.retrieval.chunks, 1):
                print(f"\n  [{i}] 来源: {chunk.source}")
                print(f"     相关性: {chunk.score}")
                print(f"     内容预览:\n{chunk.content[:200]}...")
        else:
            print("  ⚠️  未检索到任何相关片段。")

        print(f"\n{'─'*60}")
        print("📊 评估结果：")
        print(f"{'─'*60}")
        print(f"  评估结论: {result.evaluation.verdict}")
        print(f"  相关性: {result.evaluation.relevance:.2f}")
        print(f"  覆盖度: {result.evaluation.coverage:.2f}")
        print(f"  置信度: {result.evaluation.confidence:.2f}")
        print(f"  说明: {result.evaluation.conclusion}")

        print(f"\n{'─'*60}")
        print("💬 RAG 回答：")
        print(f"{'─'*60}")
        if result.answer:
            print(f"\n{result.answer}")
        else:
            print("\n❌ 无法回答该问题（知识库中未找到足够信息）")
        
        logger.info(f"[RAGDemo] RAG 问答完成  status={result.status}")
        
    except Exception as e:
        logger.exception("[RAGDemo] RAG 问答失败")
        print(f"\n❌ RAG 问答失败：{e}")


if __name__ == "__main__":
    asyncio.run(main())
