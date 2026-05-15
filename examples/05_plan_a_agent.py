"""
05_plan_a_agent.py — Plan A 完整演示：两种 RAG 模式对比

通过 ``--mode`` 选择 RAG 实现方式：

  standalone — RAGPipeline 直接检索 + LLMClient 生成回答（无需 Parlant）
  parlant    — Parlant RAG Tool 检索 + LLMClient 生成回答（**后台执行，无需 Server / UI**）
  both       — 顺序运行两种模式（默认）

所有计算/设置逻辑均在 plan_a/runner.py 中，本脚本只做初始化、调用和输出。

前置条件：
    1. 已构建知识库索引（faiss_index.bin / faiss_index + docstore）
    2. 已配置 .env（参考 .env.example）：
       - EMBEDDING_MODEL_PATH 或 EMBEDDING_BASE_URL / EMBEDDING_MODEL
       - KNOWLEDGE_BASE_PATH（或 KNOWLEDGE_PERSIST_DIR）
       - LLM 配置（QWEN3_* 或 LLM_*）
    3. 测试 query 从 knowledge_base/hotpotqa.json 中抽样读取
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from plan_a import PlanAConfig, RAGMode, RAGPipeline
from plan_a.runner import load_test_queries, run_standalone, run_parlant
from model_client.api.llm import LLMClient
from utils.logging import setup_logging


logger = setup_logging("plan_a_agent", level="INFO")
_REPO_ROOT = Path(__file__).resolve().parents[1]
_HOTPOTQA_PATH = _REPO_ROOT / "knowledge_base" / "hotpotqa.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan A 完整演示（两种 RAG 模式）")
    parser.add_argument("--limit", type=int, default=3, help="测试问题数量")
    parser.add_argument("--seed", type=int, default=0, help="抽样随机种子")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in RAGMode] + ["both"],
        default="both",
        help="standalone / parlant / both（默认 both）",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    run_standalone_mode = args.mode in (RAGMode.STANDALONE.value, "both")
    run_parlant_mode = args.mode in (RAGMode.PARLANT.value, "both")

    # --- 初始化（两种模式共用）---
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()
    await pipeline.initialize()
    queries = load_test_queries(_HOTPOTQA_PATH, limit=args.limit, seed=args.seed)

    # --- Standalone 模式 ---
    standalone_rows: list[dict] = []
    if run_standalone_mode:
        logger.info("[Entry] 运行 Standalone 模式  query_count=%d", len(queries))
        standalone_rows = await run_standalone(pipeline, queries, llm)

    # --- Parlant 模式（无 Server，后台执行）---
    parlant_rows: list[dict] = []
    parlant_status = "skipped"
    if run_parlant_mode:
        try:
            logger.info("[Entry] 运行 Parlant 模式（无 Server，后台执行）  query_count=%d", len(queries))
            parlant_rows = await run_parlant(pipeline, queries, llm)
            parlant_status = "ok"
        except ModuleNotFoundError as exc:
            if getattr(exc, "name", None) == "torch" or "No module named 'torch'" in str(exc):
                parlant_status = "missing_torch"
                logger.warning("[Entry] Parlant 启动失败：缺少 torch 依赖（pip install torch）")
            else:
                raise
        except ImportError:
            parlant_status = "missing_parlant"
            logger.warning("[Entry] parlant 包未安装，跳过（pip install parlant）")

    # --- 输出 ---
    if args.output_format == "json":
        print(json.dumps({
            "script": "05_plan_a_agent.py",
            "request": {"limit": args.limit, "seed": args.seed, "mode": args.mode},
            "standalone": standalone_rows,
            "parlant": {"status": parlant_status, "results": parlant_rows},
        }, ensure_ascii=False, indent=2))
    else:
        sep = "─" * 60
        print("=" * 60)
        print(f"  Plan A — 两种 RAG 模式演示  mode={args.mode}")
        print("=" * 60)

        if standalone_rows:
            print(f"\n[STANDALONE]  {len(standalone_rows)} 条问答")
            for row in standalone_rows:
                q = row["retrieval"]["query"]
                cnt = row["retrieval"]["chunk_count"]
                a = row["answer"][:200]
                print(f"{sep}\n  Q: {q}\n  chunks={cnt}\n  A: {a}")

        if run_parlant_mode:
            print(f"\n[PARLANT]  status={parlant_status}")
            for row in parlant_rows:
                q = row["request"].get("query", "")
                cnt = row["response"].get("chunk_count", 0)
                a = row.get("answer", "")[:200]
                print(f"{sep}\n  Q: {q}\n  chunks={cnt}\n  A: {a}")

        print(f"\n{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
