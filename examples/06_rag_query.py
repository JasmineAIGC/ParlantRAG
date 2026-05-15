"""
06_rag_query.py — Parlant Server 模式单条 RAG 查询示例

启动 Parlant Server（NLP: LiteLLM，使用 .env 配置的模型），创建 Agent + Guidelines（SR + Tool），
在后台程序化执行 query-answer，不弹出任何 UI 界面。

所有 Parlant 设置/检索逻辑均在 plan_a/runner.py 中，本脚本只做初始化和输出。
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
from plan_a.runner import run_parlant_server
from model_client.api.llm import LLMClient
from utils.logging import setup_logging


logger = setup_logging("rag_query", level="INFO")
_DEFAULT_QUERY = "what is one of the stars of The Newcomers known for?"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parlant Server 模式 RAG 单条查询示例",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=_DEFAULT_QUERY, help="检索查询文本")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument("--output-file", type=str, default=None, metavar="PATH", help="输出 JSON 文件路径")
    parser.add_argument("--agent-name", type=str, default="plan-a-rag-agent", help="Parlant Agent 名称")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()
    await pipeline.initialize()

    agent_info, results = await run_parlant_server(
        pipeline,
        [args.query],
        agent_name=args.agent_name,
        llm=llm,
    )
    result = results[0] if results else {}

    output = {
        "script": "06_rag_query.py",
        "request": {"query": args.query, "agent_name": args.agent_name},
        "parlant": {"status": "ok", "agent": agent_info},
        "result": result,
    }

    use_json = args.output_format == "json" or args.output_file is not None
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
        chunks = result.get("response", {}).get("chunk_count", 0)
        answer = result.get("answer", "")
        print(f"✅ Parlant Server 查询完成  chunks={chunks}")
        if answer:
            print(f"   回答：{answer[:300]}")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
