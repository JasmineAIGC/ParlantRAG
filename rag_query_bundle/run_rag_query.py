"""
run_rag_query.py — Parlant Server 模式单条 RAG 查询入口（后台执行，无 UI）

本目录是自包含的（含 plan_a/、model_client/、utils/），
从本目录直接运行即可，无需额外设置 PYTHONPATH：

    cd rag_query_bundle
    python run_rag_query.py --query "你的问题"

执行流程：
  1. 初始化（PlanAConfig + RAGPipeline + LLMClient，加载 FAISS 索引）
  2. RAG 执行（Parlant Server 启动 → Agent + Guidelines → RAG Tool 检索）
  3. 后处理（LLM 生成回答 → 组装输出 dict）
  4. 输出（文本摘要或完整 JSON，可保存到文件）

各阶段耗时均会打印到终端（格式：`⏱  <阶段>  took=XXXms`）。

输出 JSON 结构
--------------
::

    {
      "query":     "<查询文本>",
      "retrieval": {
        "top_k":       <实际使用的检索片段上限>,
        "chunk_count": <命中的片段数>,
        "took_ms":     <RAG 工具内部向量检索耗时（毫秒）>,
        "chunks": [
          { "content": "<片段正文>", "source": "<来源>", "score": <相似度> },
          ...
        ]
      },
      "answer": "<LLM 生成的回答>",
      "timing": {
        "init_ms":        <初始化阶段耗时（毫秒）>,
        "rag_ms":         <RAG 执行阶段耗时（毫秒）>,
        "postprocess_ms": <后处理阶段耗时（毫秒）>,
        "total_ms":       <总耗时（毫秒）>
      }
    }

详细字段说明见同目录 README.md。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# 将本脚本所在目录（rag_query_bundle/）插入 sys.path，使 plan_a/model_client/utils 可直接导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv

load_dotenv()

from plan_a import PlanAConfig, RAGPipeline
from plan_a.runner import run_parlant_server
from model_client.api.llm import LLMClient
from utils.logging import setup_logging


logger = setup_logging("rag_query", level="INFO")
_DEFAULT_QUERY = "what is one of the stars of The Newcomers known for?"


def _ms(start: float) -> int:
    """返回从 start 到现在的毫秒数（整数）。"""
    return int((time.monotonic() - start) * 1000)


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
    t_total = time.monotonic()

    # ------------------------------------------------------------------
    # 阶段 1：初始化
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()
    await pipeline.initialize()
    init_ms = _ms(t0)
    print(f"⏱  初始化完成  took={init_ms}ms")

    # ------------------------------------------------------------------
    # 阶段 2：RAG 执行（Parlant Server + Agent + Guidelines + Tool）
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    _, results = await run_parlant_server(
        pipeline,
        [args.query],
        llm=llm,
        top_k=args.top_k,
        include_metadata=args.include_metadata,
    )
    rag_ms = _ms(t0)
    print(f"⏱  RAG 执行完成  took={rag_ms}ms")

    # ------------------------------------------------------------------
    # 阶段 3：后处理（组装输出 + LLM 回答已含在 results 中）
    # ------------------------------------------------------------------
    t0 = time.monotonic()
    result = results[0] if results else {}
    resp = result.get("response", {})
    postprocess_ms = _ms(t0)
    total_ms = _ms(t_total)
    print(f"⏱  后处理完成   took={postprocess_ms}ms  |  总耗时={total_ms}ms")

    output = {
        "query": args.query,
        "retrieval": {
            "top_k": resp.get("top_k"),
            "chunk_count": resp.get("chunk_count", 0),
            "took_ms": resp.get("took_ms"),
            "chunks": resp.get("chunks", []),
        },
        "answer": result.get("answer", ""),
        "timing": {
            "init_ms": init_ms,
            "rag_ms": rag_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
        },
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
