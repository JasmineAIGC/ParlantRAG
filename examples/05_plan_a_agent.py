"""
05_plan_a_agent.py — Plan A 架构：Parlant Agent + Scoped Retriever + RAG Tool 完整演示

对应 docs/planA.md 中描述的极简 RAG 架构：

    用户 <→ Parlant Agent → Guidelines
                              ├─ Scoped Retriever（自动触发，主力方式）
                              └─ RAG Tool（手动调用，辅助方式）

运行方式：
    python examples/05_plan_a_agent.py

前置条件：
    1. 已生成知识库索引（运行 examples/01_load_knowledge_base.py）
    2. 已配置 .env（参考 .env.example）
       - LLM 配置: QWEN3_API_KEY / QWEN3_BASE_URL / QWEN3_MODEL
       - Embedding 配置: EMBEDDING_BASE_URL / EMBEDDING_MODEL
       - 知识库路径: KNOWLEDGE_BASE_PATH（或 KNOWLEDGE_PERSIST_DIR）
"""
from __future__ import annotations

import asyncio
import os
import sys

# 将项目根目录加入 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from plan_a import RagService, build_plan_a_agent
from utils.logging import setup_logging

logger = setup_logging("plan_a_agent_demo", level="INFO")


async def main() -> None:
    print(f"{'=' * 60}")
    print("🤖  Plan A — Parlant Agent + RAG 系统演示")
    print(f"{'=' * 60}")
    print()
    print("架构说明：")
    print("  📌 Scoped Retriever：Guideline 匹配时自动检索，注入上下文")
    print("  🔧 RAG Tool        ：Agent 主动调用，精准检索特定问题")
    print()

    # ------------------------------------------------------------------
    # 1. 初始化 RAG 服务
    # ------------------------------------------------------------------
    print("⏳ 初始化 RAG 服务（加载 Embedding 模型和向量索引）...")
    logger.info("[Demo] 初始化 RagService")

    rag_service = RagService()
    await rag_service.initialize()

    print("✅ RAG 服务初始化完成")
    print()

    # ------------------------------------------------------------------
    # 2. 快速验证检索效果（不启动 Parlant Server）
    # ------------------------------------------------------------------
    test_queries = [
        "30天能退货吗？",
        "会员有哪些权益？",
        "如何联系客服？",
    ]

    print(f"{'─' * 60}")
    print("🔍  独立检索测试（验证 RagService 可用性）")
    print(f"{'─' * 60}")

    for query in test_queries:
        print(f"\n问题：{query}")
        chunks = await rag_service.retrieve(query)
        if chunks:
            top = chunks[0]
            print(f"  命中 {len(chunks)} 个片段，最相关片段：")
            print(f"  来源: {top.source}  相关度: {top.score:.3f}")
            print(f"  内容预览: {top.content[:80]}...")
        else:
            print("  ⚠️  未命中（知识库为空或内容不相关）")

    print()

    # ------------------------------------------------------------------
    # 3. 启动 Parlant Server 并构建 Plan A Agent
    # ------------------------------------------------------------------
    print(f"{'─' * 60}")
    print("🚀  启动 Parlant Server，构建 Plan A Agent")
    print(f"{'─' * 60}")

    try:
        import parlant.sdk as p

        async with p.Server() as server:
            logger.info("[Demo] Parlant Server 启动")
            print("\n⏳ 构建 Plan A Agent（Agent + Guidelines + Retriever + Tool）...")

            agent = await build_plan_a_agent(server, rag_service)

            print(f"\n✅ Plan A Agent 构建完成")
            print(f"   Agent ID  : {agent.id}")
            print(f"   Agent 名称: KnowledgeAssistant")
            print()
            print("  已挂载组件：")
            print("  ✓ Scoped Retriever  — 用户询问知识库内容时自动触发")
            print("  ✓ RAG Tool          — 用户明确要求检索时由 Agent 主动调用")
            print()
            print("🟢 服务已就绪，等待用户会话...")
            logger.info("[Demo] Plan A Agent 就绪")

            # 实际部署时 server 会持续运行；演示中暂停 5 秒后退出
            await asyncio.sleep(5)
            print("\n（演示模式：5 秒后自动退出）")

    except ImportError:
        print(
            "\n⚠️  未安装 parlant 包（pip install parlant），跳过 Parlant Server 启动步骤。\n"
            "   RagService 独立检索功能已正常运行（见上方测试输出）。\n"
            "   安装 parlant 后即可完整运行 Parlant Agent。"
        )
        logger.warning("[Demo] parlant 包未安装，跳过 Server 启动")

    print(f"\n{'=' * 60}")
    print("✅  演示结束")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
