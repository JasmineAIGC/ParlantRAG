"""
03_test_retrieval_eval.py — 测试检索+LLM 评估功能

使用 plan_a.RAGPipeline 检索文档片段，再通过 model_client.LLMClient
调用 LLM 评估检索质量（相关性 / 覆盖度 / 置信度）。
测试问题从 knowledge_base/hotpotqa.json 中随机抽取。

运行方式：
    python examples/03_test_retrieval_eval.py

前置条件：
    1. 已配置 .env（KNOWLEDGE_PERSIST_DIR、EMBEDDING_*、QWEN3_* 或 LLM_*）
    2. knowledge_base/hotpotqa.json 存在
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import List

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv()

from plan_a import PlanAConfig, RAGPipeline
from plan_a.core.types import DocChunk
from model_client.api.llm import LLMClient
from utils.logging import setup_logging

logger = setup_logging("test_retrieval_eval", level="INFO", log_dir=str(_REPO_ROOT / "logs"))

_HOTPOTQA_PATH = _REPO_ROOT / "knowledge_base" / "hotpotqa.json"


def _load_test_queries(limit: int = 2) -> list[str]:
    with _HOTPOTQA_PATH.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    questions = [
        item["question"].strip()
        for item in samples
        if isinstance(item, dict) and isinstance(item.get("question"), str) and item["question"].strip()
    ]
    if not questions:
        raise ValueError(f"未从数据集读取到可用问题：{_HOTPOTQA_PATH}")
    return Random(42).sample(questions, min(limit, len(questions)))


@dataclass
class EvalResult:
    verdict: str
    relevance: float
    coverage: float
    confidence: float
    conclusion: str
    action_hint: str


async def _evaluate(llm: LLMClient, question: str, chunks: List[DocChunk]) -> EvalResult:
    """使用 LLM 评估检索结果质量。"""
    if not chunks:
        return EvalResult(
            verdict="no_coverage",
            relevance=0.0, coverage=0.0, confidence=0.0,
            conclusion="未检索到任何相关片段",
            action_hint="告知用户当前问题超出知识库覆盖范围",
        )

    context_text = "\n\n".join(
        f"[片段 {i+1}, 相关度: {c.score:.3f}]\n{c.content}"
        for i, c in enumerate(chunks)
    )
    prompt = f"""请评估以下检索结果是否能回答用户问题，输出严格的 JSON，不含其他内容：

用户问题：{question}

检索内容：
{context_text}

输出格式：
{{
  "verdict": "qualified" | "insufficient" | "no_coverage",
  "relevance": <0-1>,
  "coverage": <0-1>,
  "confidence": <0-1>,
  "conclusion": "<简要说明>",
  "action_hint": "<下一步建议>"
}}"""

    try:
        response = await llm.chat_async([{"role": "user", "content": prompt}])
        parsed = json.loads(response)
        return EvalResult(
            verdict=parsed.get("verdict", "insufficient"),
            relevance=float(parsed.get("relevance", 0.5)),
            coverage=float(parsed.get("coverage", 0.5)),
            confidence=float(parsed.get("confidence", 0.5)),
            conclusion=parsed.get("conclusion", ""),
            action_hint=parsed.get("action_hint", ""),
        )
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        logger.warning("[EvalTest] 评估解析失败: %s", exc)
        return EvalResult(
            verdict="insufficient",
            relevance=0.5, coverage=0.5, confidence=0.3,
            conclusion=f"评估失败: {exc}",
            action_hint="手动检查检索结果",
        )
    except Exception as exc:
        logger.error("[EvalTest] LLM 调用失败: %s", exc)
        return EvalResult(
            verdict="insufficient",
            relevance=0.5, coverage=0.5, confidence=0.3,
            conclusion=f"LLM 调用失败: {str(exc)[:80]}",
            action_hint="检查 LLM API 配置",
        )


async def main() -> None:
    print("=" * 60)
    print("📊  检索 + LLM 评估测试")
    print("=" * 60)
    print(f"  测试问题来源：{_HOTPOTQA_PATH}")
    logger.info("[EvalTest] 开始检索评估测试")

    config = PlanAConfig.from_env()
    pipeline = RAGPipeline(config)
    llm = LLMClient()

    print("\n⏳ 初始化 RAG 管道...")
    await pipeline.initialize()
    print("✅ RAG 管道就绪\n")

    test_queries = _load_test_queries()
    for query in test_queries:
        print(f"{'─'*60}")
        print(f"  问：{query}")
        logger.info("[EvalTest] 检索  query=%r", query)

        # 检索
        chunks = await pipeline.retrieve(query)
        print(f"  ✓ 命中 {len(chunks)} 个片段")

        # LLM 评估
        print("  ⏳ LLM 评估中...")
        result = await _evaluate(llm, query, chunks)

        print(f"\n  📊 评估结果：")
        print(f"     结论：{result.verdict}")
        print(f"     相关性：{result.relevance:.2f}  覆盖度：{result.coverage:.2f}  置信度：{result.confidence:.2f}")
        print(f"     说明：{result.conclusion}")
        print(f"     建议：{result.action_hint}")
        logger.info(
            "[EvalTest] verdict=%s  relevance=%.2f  coverage=%.2f  confidence=%.2f",
            result.verdict, result.relevance, result.coverage, result.confidence
        )

    print(f"\n{'='*60}")
    print("✅  评估测试完成")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
