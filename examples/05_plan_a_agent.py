"""
05_plan_a_agent.py — Plan A 架构：外部 Parlant Agent 入口脚本

根据 docs/planA.md，Agent[Parlant Agent] 是整个系统的外部入口，
不属于 plan_a/ RAG 功能模块。本脚本演示如何从外部创建 Agent，
并将 plan_a/ 提供的 RAG 功能挂载进去。

职责分工
--------
  本脚本（外部入口）：
    - 初始化 Parlant Server
    - 创建 Parlant Agent（name / description / composition_mode）
    - 调用 setup_guidelines() 将 RAG 模块挂载到 Agent

  plan_a/ 模块（RAG 功能，与 Agent 解耦）：
    - RAGPipeline  → 核心检索管道（Embedding + VectorStore）
    - Scoped Retriever → Guideline 匹配时自动触发检索（主力方式）
    - RAG Tool     → Agent 主动调用的检索工具（辅助方式）
    - setup_guidelines() → 将上述功能绑定到 Agent 的 Guidelines（GL 层）

运行方式：
    python examples/05_plan_a_agent.py

前置条件：
    1. 已构建知识库索引（含 faiss_index 和 document_store.json）
    2. 已配置 .env（参考 .env.example）：
       - EMBEDDING_BASE_URL / EMBEDDING_MODEL
       - KNOWLEDGE_BASE_PATH（或 KNOWLEDGE_PERSIST_DIR）
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from plan_a import PlanAConfig, RAGPipeline, setup_guidelines
from utils.logging import setup_logging


logger = setup_logging("plan_a_agent", level="INFO")

# ---------------------------------------------------------------------------
# Agent 配置（外部入口，由调用方决定，与 RAG 逻辑解耦）
# ---------------------------------------------------------------------------
_AGENT_NAME = "KnowledgeAssistant"
_AGENT_DESCRIPTION = (
    "专业的知识库问答助手。"
    "当用户提出问题时，会自动检索相关知识库内容并给出准确、简洁的回答。"
    "如果知识库中没有相关内容，会如实告知。"
)


async def main() -> None:
    print("=" * 60)
    print("🤖  Plan A — Parlant Agent + RAG 系统")
    print("    Agent 是外部入口，RAG 功能由 plan_a/ 模块提供")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 初始化 RAG 管道（独立于 Agent，可在 Agent 启动前完成）
    # ------------------------------------------------------------------
    print("\n⏳ 初始化 RAG 管道（加载 Embedding + FAISS 索引）...")
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    await pipeline.initialize()
    print("✅ RAG 管道初始化完成")

    # ------------------------------------------------------------------
    # 2. 快速验证 RAG 管道的独立检索能力（无需 Agent）
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("🔍  RAG 管道独立检索测试")
    print(f"{'─' * 60}")
    test_queries = [
        "30天能退货吗？",
        "会员有哪些权益？",
        "如何联系客服？",
    ]
    for query in test_queries:
        print(f"\n  问：{query}")
        chunks = await pipeline.retrieve(query)
        if chunks:
            top = chunks[0]
            print(f"  ✓ 命中 {len(chunks)} 片段  最相关: {top.source} ({top.score:.3f})")
            print(f"    {top.content[:80]}...")
        else:
            print("  ⚠️  未命中（知识库为空或内容不相关）")

    # ------------------------------------------------------------------
    # 3. 创建 Parlant Agent 并挂载 RAG 功能（Agent 是外部入口）
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("🚀  启动 Parlant Server，创建外部 Agent 并挂载 RAG 功能")
    print(f"{'─' * 60}")

    try:
        import parlant.sdk as p

        async with p.Server() as server:
            logger.info("[Entry] Parlant Server 启动")

            # ─ Agent 创建（外部入口，不在 plan_a/ 模块内）─
            print(f"\n⏳ 创建 Agent: {_AGENT_NAME!r}")
            agent = await server.create_agent(
                name=_AGENT_NAME,
                description=_AGENT_DESCRIPTION,
            )
            print(f"✅ Agent 创建完成  id={agent.id}")

            # ─ 挂载 RAG 功能（调用 plan_a/guidelines/ 下的 setup_guidelines）─
            print("\n⏳ 配置 Guidelines（Scoped Retriever + RAG Tool）...")
            scoped_gl, tool_gl = await setup_guidelines(agent, pipeline)
            print("✅ Guidelines 配置完成")
            print(f"   Scoped Retriever GL : {scoped_gl.id}")
            print(f"   RAG Tool GL         : {tool_gl.id}")

            print("\n🟢 服务就绪，等待用户会话...")
            print("   SR  → Guideline 匹配时自动检索知识库（主力方式）")
            print("   Tool→ 用户明确要求检索时由 Agent 调用（辅助方式）")
            logger.info(f"[Entry] Plan A Agent 就绪  agent_id={agent.id}")

            # 演示模式：等待 5 秒后退出
            await asyncio.sleep(5)
            print("\n（演示模式：5 秒后自动退出）")

    except ImportError:
        print(
            "\n⚠️  未安装 parlant（pip install parlant），跳过 Agent 启动。\n"
            "   plan_a/ RAG 管道的独立检索功能已正常演示（见上方输出）。"
        )
        logger.warning("[Entry] parlant 包未安装，跳过 Server 启动")

    print(f"\n{'=' * 60}")
    print("✅  演示结束")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
