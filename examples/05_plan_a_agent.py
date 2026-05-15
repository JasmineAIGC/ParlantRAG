"""
05_plan_a_agent.py — Plan A 完整演示：独立 RAG 问答 + Parlant RAG 模块验证

本脚本分三个阶段演示 plan_a/ 模块的完整能力：

阶段一：初始化 RAG 管道
  加载 Embedding 模型 + FAISS 向量索引。

阶段二：独立 RAG 问答（无需 Parlant Server）
  检索（plan_a.RAGPipeline）→ LLM 回答（model_client.LLMClient）
  对应 RAGMode.STANDALONE。

阶段三：Parlant RAG 模块验证（无需 Server / UI）
  直接创建 RAG Tool 与 Scoped Retriever，验证 @p.tool 注册与检索逻辑，
  无需启动 Parlant Server 或 UI 界面。
  对应 RAGMode.PARLANT。

职责分工
--------
  plan_a/ 模块（RAG 功能，与 Agent 解耦）：
    - RAGPipeline  → 核心检索管道（Embedding + VectorStore）
    - Scoped Retriever → Guideline 匹配时自动触发检索（主力方式）
    - RAG Tool     → Agent 主动调用的检索工具（辅助方式）
    - setup_guidelines() → 将上述功能绑定到 Agent 的 Guidelines（GL 层）

运行方式：
    # 默认：同时运行两种方式
    python examples/05_plan_a_agent.py

    # 仅独立 RAG（无需 Parlant）
    python examples/05_plan_a_agent.py --mode standalone

    # 仅 Parlant RAG 模块验证
    python examples/05_plan_a_agent.py --mode parlant

前置条件：
    1. 已构建知识库索引（支持 faiss_index.bin / faiss_index 及不同文档持久化格式）
    2. 已配置 .env（参考 .env.example）：
        - EMBEDDING_MODEL_PATH 或 EMBEDDING_BASE_URL / EMBEDDING_MODEL
        - KNOWLEDGE_BASE_PATH（或 KNOWLEDGE_PERSIST_DIR）
        - LLM 相关配置（QWEN3_* 或 LLM_*；会自动桥接为 Parlant 所需的 LITELLM_PROVIDER_*）
    3. 测试 query 从 knowledge_base/hotpotqa.json 中抽样读取
"""
from __future__ import annotations

import argparse
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
from plan_a import PlanAConfig, RAGMode, RAGPipeline
from plan_a.core.types import DocChunk
from plan_a.core import RetrievalRequest
from model_client.api.llm import LLMClient
from utils.logging import setup_logging


logger = setup_logging("plan_a_agent", level="INFO")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HOTPOTQA_PATH = _REPO_ROOT / "knowledge_base" / "hotpotqa.json"


def _load_test_queries(limit: int = 3, seed: int = 0) -> list[str]:
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
    return Random(seed).sample(questions, limit)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan A 完整演示")
    parser.add_argument("--limit", type=int, default=3, help="测试问题数量")
    parser.add_argument("--seed", type=int, default=0, help="抽样随机种子")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in RAGMode] + ["both"],
        default="both",
        help=(
            "RAG 实现方式：\n"
            "  standalone — 不依赖 Parlant，直接使用 RAGPipeline 检索（阶段二）\n"
            "  parlant    — 依赖 Parlant，验证 SR / GL / Tool 三件套（阶段三）\n"
            "  both       — 同时运行两种方式（默认）"
        ),
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    run_standalone = args.mode in (RAGMode.STANDALONE.value, "both")
    run_parlant = args.mode in (RAGMode.PARLANT.value, "both")

    if args.output_format == "text":
        print("=" * 60)
        print("🤖  Plan A — 独立 RAG 问答 + Parlant Agent 集成演示")
        print("    检索：plan_a.RAGPipeline  |  回答：model_client.LLMClient")
        print(f"    模式：{args.mode}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # 1. 初始化 RAG 管道（独立于 Agent，可在 Agent 启动前完成）
    # ------------------------------------------------------------------
    if args.output_format == "text":
        print("\n⏳ 初始化 RAG 管道（加载 Embedding + FAISS 索引）...")
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()
    await pipeline.initialize()
    if args.output_format == "text":
        print("✅ RAG 管道初始化完成")

    # ------------------------------------------------------------------
    # 2. 快速验证 RAG 管道的独立检索 + LLM 问答能力（无需 Agent）
    #    对应 RAGMode.STANDALONE
    # ------------------------------------------------------------------
    qa_rows: list[dict] = []
    if run_standalone:
        if args.output_format == "text":
            print(f"\n{'─' * 60}")
            print("🔍  RAG 问答演示（RAGMode.STANDALONE — 无需 Parlant Server）")
            print(f"{'─' * 60}")
        test_queries = _load_test_queries(limit=args.limit, seed=args.seed)
        if args.output_format == "text":
            print(f"📚 测试问题来源：{_HOTPOTQA_PATH}\n")
        for query in test_queries:
            if args.output_format == "text":
                print(f"{'─'*60}")
                print(f"  问：{query}")
            logger.info("[PlanAAgent] query=%r", query)

            retrieval_response = await pipeline.retrieve_standard(RetrievalRequest(query=query))
            chunks = retrieval_response.chunks
            if chunks:
                top = chunks[0]
                # 优先使用 metadata 中的标题；source 在部分格式下为 UUID
                title = top.metadata.get("title") or top.source
                if args.output_format == "text":
                    print(f"  ✓ 命中 {len(chunks)} 片段  最相关: {title} ({top.score:.3f})")
            else:
                if args.output_format == "text":
                    print("  ⚠️  未命中（知识库为空或内容不相关）")

            if args.output_format == "text":
                print("  ⏳ LLM 生成回答中...")
            answer = await _answer(llm, query, chunks)
            qa_rows.append({
                "retrieval": retrieval_response.to_dict(include_metadata=False),
                "answer": answer.strip(),
            })
            if args.output_format == "text":
                print(f"\n  💬 回答：")
                print(f"     {answer.strip()[:300]}")
            logger.info("[PlanAAgent] 问答完成  chunk_count=%d", len(chunks))
    else:
        # 需要 test_queries 供阶段三使用
        test_queries = _load_test_queries(limit=args.limit, seed=args.seed)

    # ------------------------------------------------------------------
    # 3. Parlant RAG 模块验证（无需 Server / UI，代码直接调用）
    #    对应 RAGMode.PARLANT
    # ------------------------------------------------------------------
    parlant_rows: list[dict] = []
    parlant_status = "skipped"

    if run_parlant:
        if args.output_format == "text":
            print(f"\n{'─' * 60}")
            print("🔧  Parlant RAG 模块验证（RAGMode.PARLANT — 代码直接调用，无需 Server）")
            print(f"{'─' * 60}")

        try:
            import parlant.sdk as p

            parlant_env = bridge_parlant_litellm_env()
            logger.info(
                "[Entry] Parlant NLP 环境已准备  model=%s  base_url=%s",
                parlant_env.get("LITELLM_PROVIDER_MODEL_NAME"),
                parlant_env.get("LITELLM_PROVIDER_BASE_URL"),
            )

            from plan_a.tools import make_rag_tool
            from plan_a.retrievers import make_scoped_retriever

            # ─ 创建 RAG Tool（无需运行 Agent / Server）─
            search_knowledge = make_rag_tool(pipeline)
            if args.output_format == "text":
                print("✅ RAG Tool 创建成功（@p.tool 注册通过）")

            # ─ 创建 Scoped Retriever（无需运行 Agent / Server）─
            retriever_fn = make_scoped_retriever(pipeline)
            if args.output_format == "text":
                print("✅ Scoped Retriever 创建成功")

            # ─ 直接调用 RAG 管道验证工具逻辑 ─
            if args.output_format == "text":
                print("\n⏳ 直接调用 RAG 工具函数验证检索效果...")
            for query in test_queries[:2]:
                if args.output_format == "text":
                    print(f"\n{'─'*60}")
                    print(f"  问：{query}")
                logger.info("[Entry] RAG Tool 直接调用  query=%r", query)

                tool_response = await pipeline.retrieve_standard(RetrievalRequest(query=query))
                chunks = tool_response.chunks
                parlant_rows.append(tool_response.to_dict(include_metadata=False))
                if chunks:
                    top = chunks[0]
                    title = top.metadata.get("title") or top.source
                    if args.output_format == "text":
                        print(f"  ✓ 工具命中 {len(chunks)} 片段  最相关: {title} ({top.score:.3f})")
                else:
                    if args.output_format == "text":
                        print("  ⚠️  工具未命中（知识库为空或内容不相关）")

                logger.info("[Entry] RAG Tool 调用完成  chunk_count=%d", len(chunks))

            logger.info("[Entry] Parlant RAG 模块验证完成")
            parlant_status = "ok"
            if args.output_format == "text":
                print("\n✅ Parlant RAG 模块验证通过（无需 UI / Server）")

        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "torch" or "No module named 'torch'" in str(exc):
                parlant_status = "missing_torch"
                if args.output_format == "text":
                    print(
                        "\n⚠️  Parlant LiteLLM 适配器依赖 `torch`，当前环境缺失。\n"
                        "   请先安装 torch（例如 `pip install torch`），再重试。"
                    )
                logger.warning("[Entry] Parlant 启动失败：缺少 torch 依赖")
            else:
                raise
        except ImportError:
            parlant_status = "missing_parlant"
            if args.output_format == "text":
                print(
                    "\n⚠️  未安装 parlant（pip install parlant），跳过 RAG 模块验证。\n"
                    "   plan_a/ RAG 管道的独立检索功能已正常演示（见上方输出）。"
                )
            logger.warning("[Entry] parlant 包未安装，跳过模块验证")

    if args.output_format == "json":
        print(json.dumps({
            "script": "05_plan_a_agent.py",
            "request": {"limit": args.limit, "seed": args.seed, "mode": args.mode},
            "qa": qa_rows,
            "parlant": {
                "status": parlant_status,
                "responses": parlant_rows,
            },
        }, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print("✅  演示结束")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
