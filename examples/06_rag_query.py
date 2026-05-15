"""
06_rag_query.py — 基于 plan_a/ 的 RAG 检索调用示例

本脚本演示如何以最少代码调用 plan_a/ 模块提供的 RAG 检索能力。
适合初次接入 RAG 功能的开发者参考。

功能说明
--------
1. 初始化 RAG 管道（Embedding + FAISS 索引）。
2. 使用 :class:`~plan_a.core.RetrievalRequest` 构造标准化检索请求，
   通过 :meth:`~plan_a.rag.RAGPipeline.retrieve_standard` 获取
   :class:`~plan_a.core.RetrievalResponse` 标准化检索结果。
3. 可选：通过 Parlant ``make_rag_tool`` 直接调用 RAG Tool（验证 Parlant 集成）。
4. 支持 ``--output-format json`` 将结果打印为 JSON，
   或 ``--output-file <path>`` 将 JSON 结果保存到指定文件。

运行示例
--------
::

    # 使用默认 query，文本输出
    python examples/06_rag_query.py

    # 自定义 query 与 top-k，JSON 输出
    python examples/06_rag_query.py \\
        --query "What is known for one of the stars of The Newcomers?" \\
        --top-k 5 \\
        --include-metadata \\
        --output-format json

    # 将结果保存为文件
    python examples/06_rag_query.py --output-file results/rag_output.json

前置条件
--------
1. 已构建知识库索引（faiss_index.bin 或 faiss_index）。
2. 已配置 ``.env``：
   - ``EMBEDDING_MODEL_PATH`` 或 ``EMBEDDING_BASE_URL`` / ``EMBEDDING_MODEL``
   - ``KNOWLEDGE_BASE_PATH``（或 ``KNOWLEDGE_PERSIST_DIR``）
   - LLM 相关配置（``QWEN3_*`` 或 ``LLM_*``）
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

from plan_a import PlanAConfig, RAGPipeline
from plan_a.core import RetrievalRequest, RetrievalResponse
from utils.logging import setup_logging

logger = setup_logging("06_rag_query", level="INFO")

_DEFAULT_QUERY = "what is one of the stars of  The Newcomers known for？"


# ---------------------------------------------------------------------------
# CLI 参数解析
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="plan_a RAG 检索示例（带 Parlant Tool 验证）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--query",
        type=str,
        default=_DEFAULT_QUERY,
        help=f"检索查询文本（默认：{_DEFAULT_QUERY!r}）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="K",
        help="返回片段数量上限（默认：使用配置文件中的 top_k 值）",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        default=False,
        help="在检索结果中包含文档元数据（title、url 等）",
    )
    parser.add_argument(
        "--skip-parlant",
        action="store_true",
        default=False,
        help="跳过 Parlant RAG Tool 验证步骤",
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="输出格式：text（默认）或 json",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        metavar="PATH",
        help="将 JSON 结果写入指定文件（同时隐含 --output-format json）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

async def run_retrieval(
    pipeline: RAGPipeline,
    query: str,
    top_k: int | None,
    include_metadata: bool,
) -> RetrievalResponse:
    """构造 RetrievalRequest 并执行标准化检索。"""
    request = RetrievalRequest(
        query=query,
        top_k=top_k,
        include_metadata=include_metadata,
    )
    return await pipeline.retrieve_standard(request)


async def run_parlant_tool(
    pipeline: RAGPipeline,
    query: str,
) -> dict:
    """
    通过 Parlant make_rag_tool 验证 RAG Tool 集成。

    Returns:
        描述验证结果的字典（status、tool_result 或 error_type）。
    """
    try:
        from model_client._utils import bridge_parlant_litellm_env
        from plan_a.tools import make_rag_tool

        bridge_parlant_litellm_env()
        search_knowledge = make_rag_tool(pipeline)
        tool_response = await pipeline.retrieve_standard(RetrievalRequest(query=query))
        return {
            "status": "ok",
            "tool_name": search_knowledge.__ne__,
            "tool_result": tool_response.to_dict(include_metadata=False),
        }
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "torch" or "No module named 'torch'" in str(exc):
            return {"status": "missing_torch", "error": str(exc)}
        raise
    except ImportError as exc:
        return {"status": "missing_parlant", "error": str(exc)}


# ---------------------------------------------------------------------------
# 输出格式化
# ---------------------------------------------------------------------------

def _print_response_text(response: RetrievalResponse, include_metadata: bool) -> None:
    """文本格式输出检索结果。"""
    print(f"\n{'─' * 60}")
    print(f"  查询：{response.query}")
    print(f"  命中：{len(response.chunks)} 个片段  top_k={response.top_k}  耗时={response.took_ms}ms")
    print(f"{'─' * 60}")
    if not response.chunks:
        print("  ⚠️  未找到相关内容（知识库为空或内容不相关）")
        return
    for i, chunk in enumerate(response.chunks, 1):
        title = chunk.metadata.get("title") or chunk.source
        print(f"\n  [{i}] {title}  (score={chunk.score:.4f})")
        print(f"      {chunk.content[:200].replace(chr(10), ' ')}")
        if include_metadata and chunk.metadata:
            print(f"      元数据：{chunk.metadata}")


def _print_parlant_text(parlant_result: dict) -> None:
    """文本格式输出 Parlant Tool 验证结果。"""
    status = parlant_result.get("status", "unknown")
    if status == "ok":
        tr = parlant_result.get("tool_result", {})
        chunk_count = tr.get("chunk_count", 0)
        print(f"✅ Parlant RAG Tool（{parlant_result.get('tool_name', '?')}）验证通过  命中={chunk_count} 片段")
    elif status == "missing_torch":
        print("⚠️  Parlant 依赖 torch 但未安装，跳过 Tool 验证")
    elif status == "missing_parlant":
        print("⚠️  parlant 包未安装（pip install parlant），跳过 Tool 验证")
    else:
        print(f"⚠️  Parlant 验证状态：{status}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    use_json = args.output_format == "json" or args.output_file is not None

    if not use_json:
        print("=" * 60)
        print("🔍  RAG 检索示例 — plan_a/ + Parlant 集成验证")
        print("=" * 60)

    # -- 初始化 RAG 管道 --
    if not use_json:
        print("\n⏳ 初始化 RAG 管道（加载 Embedding + FAISS 索引）...")

    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    await pipeline.initialize()

    if not use_json:
        print("✅ RAG 管道初始化完成")

    # -- 执行检索 --
    if not use_json:
        print(f"\n{'─' * 60}")
        print("🔎  执行 RAG 检索")
        print(f"{'─' * 60}")
        print(f"  查询文本：{args.query!r}")
        if args.top_k:
            print(f"  top_k   ：{args.top_k}")
        print(f"  元数据  ：{'包含' if args.include_metadata else '不包含'}")

    response = await run_retrieval(
        pipeline,
        query=args.query,
        top_k=args.top_k,
        include_metadata=args.include_metadata,
    )
    logger.info("[06_rag_query] 检索完成  query=%r  chunk_count=%d  took_ms=%d",
                args.query, len(response.chunks), response.took_ms)

    if not use_json:
        _print_response_text(response, include_metadata=args.include_metadata)

    # -- Parlant Tool 验证 --
    parlant_result: dict = {"status": "skipped"}
    if not args.skip_parlant:
        if not use_json:
            print(f"\n{'─' * 60}")
            print("🔧  Parlant RAG Tool 验证")
            print(f"{'─' * 60}")
        parlant_result = await run_parlant_tool(pipeline, query=args.query)
        if not use_json:
            _print_parlant_text(parlant_result)

    # -- 构建输出字典 --
    output = {
        "script": "06_rag_query.py",
        "request": {
            "query": args.query,
            "top_k": args.top_k,
            "include_metadata": args.include_metadata,
        },
        "retrieval": response.to_dict(include_metadata=args.include_metadata),
        "parlant_tool": parlant_result,
    }

    # -- 输出 JSON --
    if use_json:
        json_text = json.dumps(output, ensure_ascii=False, indent=2)
        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_text, encoding="utf-8")
            print(f"✅ 结果已保存至：{out_path}")
        else:
            print(json_text)
    else:
        print(f"\n{'=' * 60}")
        print("✅  检索演示结束")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
