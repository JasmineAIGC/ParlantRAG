"""
run_rag_query.py — Parlant Server 模式单条 RAG 查询入口（后台执行，无 UI）

通过完整的 Parlant Agent + Guidelines + RAG Tool 执行单条 query-answer：
  1. 启动 Parlant Server（后台，不弹出 UI）
  2. 创建 Agent，并配置 Scoped Retriever 和 RAG Tool 两条 Guidelines
  3. 执行检索并用 LLM 生成回答
  4. 输出结构化结果

所有检索/设置逻辑均在 plan_a/runner.py 中，本脚本只做初始化和输出。

输出 JSON 结构
--------------
::

    {
      "query":     "<查询文本>",
      "retrieval": {
        "top_k":       <实际使用的检索片段上限>,
        "chunk_count": <命中的片段数>,
        "took_ms":     <检索耗时（毫秒）>,
        "chunks": [
          { "content": "<片段正文>", "source": "<来源>", "score": <相似度> },
          ...
        ]
      },
      "answer": "<LLM 生成的回答>"
    }

详细字段说明见同目录 README.md。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# 将仓库根目录加入 sys.path，保证可以 import plan_a / model_client / utils
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
        description="Parlant 模式 RAG 单条查询（后台执行，无 UI）",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--query", type=str, default=_DEFAULT_QUERY, help="检索查询文本")
    parser.add_argument("--top-k", type=int, default=None, metavar="K", help="检索返回的最大片段数（默认使用 pipeline 配置）")
    parser.add_argument("--include-metadata", action="store_true", help="在检索结果中包含文档元数据")
    parser.add_argument("--output-format", choices=["text", "json"], default="text", help="输出格式")
    parser.add_argument("--output-file", type=str, default=None, metavar="PATH", help="输出 JSON 文件路径")
    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()
    await pipeline.initialize()

    _, results = await run_parlant_server(
        pipeline,
        [args.query],
        llm=llm,
        top_k=args.top_k,
        include_metadata=args.include_metadata,
    )
    result = results[0] if results else {}
    resp = result.get("response", {})
    output = {
        "query": args.query,
        "retrieval": {
            "top_k": resp.get("top_k"),
            "chunk_count": resp.get("chunk_count", 0),
            "took_ms": resp.get("took_ms"),
            "chunks": resp.get("chunks", []),
        },
        "answer": result.get("answer", ""),
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
        chunks = output["retrieval"]["chunk_count"]
        answer = output["answer"]
        print(f"✅ 查询完成  chunks={chunks}")
        if answer:
            print(f"   回答：{answer[:300]}")


if __name__ == "__main__":
    asyncio.run(main(_parse_args()))
