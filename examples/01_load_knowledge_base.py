"""
load_knowledge_base.py — 测试加载知识库

验证 FAISS 索引文件能否被正常加载。

运行方式：
    python examples/load_knowledge_base.py
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
logger = setup_logging("load_knowledge_base", level="INFO", log_dir="../logs")


async def main() -> None:
    persist_dir = os.getenv("KNOWLEDGE_PERSIST_DIR", "./knowledge_base")

    print(f"{'='*60}")
    print(f"📂 知识库路径：{persist_dir}")
    print(f"{'='*60}")
    logger.info(f"[LoadKB] 开始加载知识库: {persist_dir}")

    # 检查知识库目录是否存在
    if not os.path.exists(persist_dir):
        print(f"\n❌ 未找到知识库目录：{persist_dir}")
        print("   请先构建知识库。")
        logger.error(f"[LoadKB] 知识库目录不存在: {persist_dir}")
        return

    print(f"\n✅ 知识库目录已找到：{persist_dir}")

    # 加载知识库
    print("\n⏳ 正在初始化 RAG 模块...")
    rag = RAG()
    await rag.initialize()

    print(f"\n✅ RAG 模块初始化成功")
    logger.info("[LoadKB] RAG 模块初始化成功")


if __name__ == "__main__":
    asyncio.run(main())
