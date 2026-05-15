"""
06_rag_query.py — 基于 Parlant Agent 的 RAG 查询示例（Parlant-first）

本脚本以 Parlant 为主入口，按 GL -> SR / Tool -> RAGPipeline 的方式接入 RAG。

流程说明
--------
1. 初始化 plan_a.RAGPipeline（Embedding + FAISS 索引）。
2. 创建 Parlant Agent。
3. 调用 setup_guidelines(agent, pipeline) 配置两条 Guideline：
   - GL -> SR（Scoped Retriever，自动检索）
   - GL -> Tool（RAG Tool，主动调用）
4. 通过 Parlant @p.tool 路径执行 RAG 查询（Tool 内部调用 RAGPipeline）。
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from model_client._utils import bridge_parlant_litellm_env
from plan_a import PlanAConfig, RAGPipeline
from plan_a.guidelines import setup_guidelines
from plan_a.tools import make_rag_tool

_DEFAULT_QUERY = "what is one of the stars of  The Newcomers known for？"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parlant-first RAG 查询示例（GL + SR + Tool）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=_DEFAULT_QUERY, help="检索查询文本")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="返回片段数量上限")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument("--output-file", type=str, default=None, metavar="PATH", help="输出 JSON 文件")
    parser.add_argument("--agent-name", type=str, default="plan-a-rag-query-agent", help="Parlant Agent 名称")
    return parser.parse_args()


def _trim_tool_chunks(tool_payload: dict, top_k: int | None) -> dict:
    response = dict(tool_payload)
    chunks = list(response.get("chunks") or [])
    if top_k is not None:
        chunks = chunks[:top_k]
    response["chunks"] = chunks
    response["chunk_count"] = len(chunks)
    if top_k is not None:
        response["top_k"] = top_k
    return response


async def _run_via_parlant_tool(search_knowledge, query: str, top_k: int | None) -> dict:
    result = await search_knowledge(None, query)
    data = getattr(result, "data", {}) if result is not None else {}
    response = _trim_tool_chunks(data.get("response") or {}, top_k)
    return {
        "request": data.get("request") or {"query": query, "top_k": top_k},
        "response": response,
    }


async def main(args: argparse.Namespace) -> None:
    use_json = args.output_format == "json" or args.output_file is not None

    try:
        import parlant.sdk as p
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) == "torch" or "No module named 'torch'" in str(exc):
            raise RuntimeError("Parlant 依赖 torch，请先安装：pip install torch") from exc
        raise RuntimeError("未安装 parlant，请先安装：pip install parlant") from exc

    bridge_parlant_litellm_env()
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    await pipeline.initialize()

    async with p.Server() as server:
        agent = await server.create_agent(name=args.agent_name, description="Parlant-first RAG query demo")
        scoped_guideline, rag_tool_guideline = await setup_guidelines(agent, pipeline)
        search_knowledge = make_rag_tool(pipeline)
        tool_execution = await _run_via_parlant_tool(search_knowledge, args.query, args.top_k)

    output = {
        "script": "06_rag_query.py",
        "request": {"query": args.query, "top_k": args.top_k, "agent_name": args.agent_name},
        "parlant": {
            "status": "ok",
            "agent": {"id": agent.id, "name": args.agent_name},
            "guidelines": {
                "scoped_retriever_guideline_id": scoped_guideline.id,
                "rag_tool_guideline_id": rag_tool_guideline.id,
            },
        },
        "rag_tool_execution": tool_execution,
    }

    if use_json:
        text = json.dumps(output, ensure_ascii=False, indent=2)
        if args.output_file:
            out_path = Path(args.output_file)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            print(f"✅ 结果已保存至：{out_path}")
        else:
            print(text)
    else:
        chunks = tool_execution.get("response", {}).get("chunks", [])
        print(f"✅ Parlant-first 查询完成，Tool 返回 {len(chunks)} 个片段")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
