"""
05_plan_a_agent.py — Plan A 完整演示：独立 RAG 问答 + Parlant Agent 集成

本脚本分两个阶段演示 plan_a/ 模块的完整能力：

阶段一：独立 RAG 问答（无需 Parlant Server）
  检索（plan_a.RAGPipeline）→ LLM 回答（model_client.LLMClient）

阶段二：Parlant Agent 集成
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
    1. 已构建知识库索引（支持 faiss_index.bin / faiss_index 及不同文档持久化格式）
    2. 已配置 .env（参考 .env.example）：
        - EMBEDDING_MODEL_PATH 或 EMBEDDING_BASE_URL / EMBEDDING_MODEL
        - KNOWLEDGE_BASE_PATH（或 KNOWLEDGE_PERSIST_DIR）
        - LLM 相关配置（QWEN3_* 或 LLM_*；会自动桥接为 Parlant 所需的 LITELLM_PROVIDER_*）
    3. 测试 query 从 knowledge_base/hotpotqa.json 中抽样读取
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from model_client._utils import bridge_parlant_litellm_env
from plan_a import PlanAConfig, RAGPipeline, setup_guidelines
from plan_a.core.types import DocChunk
from model_client.api.llm import LLMClient
from utils.logging import setup_logging


logger = setup_logging("plan_a_agent", level="INFO")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HOTPOTQA_PATH = _REPO_ROOT / "knowledge_base" / "hotpotqa.json"

# ---------------------------------------------------------------------------
# Agent 配置（外部入口，由调用方决定，与 RAG 逻辑解耦）
# ---------------------------------------------------------------------------
_AGENT_NAME = "KnowledgeAssistant"
_AGENT_DESCRIPTION = (
    "专业的知识库问答助手。"
    "当用户提出问题时，会自动检索相关知识库内容并给出准确、简洁的回答。"
    "如果知识库中没有相关内容，会如实告知。"
)


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

    if len(questions) <= limit:
        return questions

    # 固定 seed，保证示例输出可复现，便于对比不同改动的检索效果。
    return Random(0).sample(questions, limit)


async def _answer(llm: LLMClient, question: str, chunks: list[DocChunk]) -> str:
    """根据检索到的文档片段使用 LLM 生成回答。"""
    if not chunks:
        return "知识库中未找到与该问题相关的内容，无法回答。"

    context_text = "\n\n".join(
        f"[来源: {c.metadata.get('title') or c.source}]\n{c.content}"
        for c in chunks
    )
    prompt = (
        "请根据以下知识库内容简洁回答用户问题。若内容不足以回答，请如实说明。\n\n"
        f"用户问题：{question}\n\n"
        f"知识库内容：\n{context_text}\n\n回答："
    )
    try:
        return await llm.chat_async([{"role": "user", "content": prompt}])
    except Exception as exc:
        logger.error("[PlanAAgent] LLM 调用失败: %s", exc)
        return f"LLM 调用失败，无法生成回答：{exc}"


async def main() -> None:
    print("=" * 60)
    print("🤖  Plan A — 独立 RAG 问答 + Parlant Agent 集成演示")
    print("    检索：plan_a.RAGPipeline  |  回答：model_client.LLMClient")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 初始化 RAG 管道（独立于 Agent，可在 Agent 启动前完成）
    # ------------------------------------------------------------------
    print("\n⏳ 初始化 RAG 管道（加载 Embedding + FAISS 索引）...")
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()
    await pipeline.initialize()
    print("✅ RAG 管道初始化完成")

    # ------------------------------------------------------------------
    # 2. 快速验证 RAG 管道的独立检索 + LLM 问答能力（无需 Agent）
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("🔍  RAG 问答演示（无需 Parlant Server）")
    print(f"{'─' * 60}")
    test_queries = _load_test_queries()
    print(f"📚 测试问题来源：{_HOTPOTQA_PATH}\n")
    for query in test_queries:
        print(f"{'─'*60}")
        print(f"  问：{query}")
        logger.info("[PlanAAgent] query=%r", query)

        chunks = await pipeline.retrieve(query)
        if chunks:
            top = chunks[0]
            # 优先使用 metadata 中的标题；source 在部分格式下为 UUID
            title = top.metadata.get("title") or top.source
            print(f"  ✓ 命中 {len(chunks)} 片段  最相关: {title} ({top.score:.3f})")
        else:
            print("  ⚠️  未命中（知识库为空或内容不相关）")

        print("  ⏳ LLM 生成回答中...")
        answer = await _answer(llm, query, chunks)
        print(f"\n  💬 回答：")
        print(f"     {answer.strip()[:300]}")
        logger.info("[PlanAAgent] 问答完成  chunk_count=%d", len(chunks))

    # ------------------------------------------------------------------
    # 3. 创建 Parlant Agent 并挂载 RAG 功能（Agent 是外部入口）
    # ------------------------------------------------------------------
    print(f"\n{'─' * 60}")
    print("🚀  启动 Parlant Server，创建外部 Agent 并挂载 RAG 功能")
    print(f"{'─' * 60}")

    try:
        import parlant.sdk as p

        parlant_env = bridge_parlant_litellm_env()
        logger.info(
            "[Entry] Parlant NLP 环境已准备  model=%s  base_url=%s",
            parlant_env.get("LITELLM_PROVIDER_MODEL_NAME"),
            parlant_env.get("LITELLM_PROVIDER_BASE_URL"),
        )

        async with p.Server(nlp_service=p.NLPServices.litellm) as server:
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
    except SystemExit as exc:
        print(
            "\n⚠️  Parlant Server 启动失败。当前示例使用 LiteLLM 作为 Parlant NLP 后端。\n"
            "   请检查 .env 中的 QWEN3_MODEL / QWEN3_BASE_URL / QWEN3_API_KEY\n"
            "   （或对应的 LLM_MODEL / LLM_BASE_URL / LLM_API_KEY），\n"
            "   程序会自动桥接为 Parlant 所需的 LITELLM_PROVIDER_* 环境变量。"
        )
        logger.warning(f"[Entry] Parlant Server 启动失败，SystemExit code={exc.code!r}")

    print(f"\n{'=' * 60}")
    print("✅  演示结束")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
