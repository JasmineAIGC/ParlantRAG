"""
01_load_knowledge_base.py — 验证知识库能否被正常加载

使用 plan_a.RAGPipeline 加载本地 FAISS 索引，验证知识库文件完整性。

运行方式：
    python examples/01_load_knowledge_base.py

前置条件：
    1. 已配置 .env（KNOWLEDGE_PERSIST_DIR 或 KNOWLEDGE_BASE_PATH）
    2. 知识库目录中含有 faiss_index.bin / docstore.json 等文件
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from plan_a import PlanAConfig, RAGPipeline
from utils.logging import setup_logging

logger = setup_logging("load_knowledge_base", level="INFO", log_dir=str(_REPO_ROOT / "logs"))


async def main() -> None:
    config = PlanAConfig.from_env()
    persist_dir = config.knowledge_base_path

    print("=" * 60)
    print("📂  知识库加载验证")
    print("=" * 60)
    print(f"  知识库路径：{persist_dir}")
    logger.info("[LoadKB] 开始加载知识库  path=%s", persist_dir)

    if not os.path.exists(persist_dir):
        print(f"\n❌ 知识库目录不存在：{persist_dir}")
        print("   请先构建知识库或检查 .env 中 KNOWLEDGE_PERSIST_DIR 的配置。")
        logger.error("[LoadKB] 知识库目录不存在: %s", persist_dir)
        return

    print(f"\n⏳ 正在初始化 RAG 管道（加载 FAISS 索引）...")
    try:
        pipeline = RAGPipeline(config)
        await pipeline.initialize()
        print("✅ 知识库加载成功")
        logger.info("[LoadKB] 知识库加载成功")
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        print(f"\n❌ 知识库加载失败：{exc}")
        logger.error("[LoadKB] 知识库加载失败: %s", exc)
        return

    print(f"\n{'='*60}")
    print("✅  验证通过，知识库可正常使用")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
