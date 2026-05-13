"""
LLMEvaluator — 多信号融合 + LLM 综合评估器。

评估流程
--------
1. 本地预计算量化信号（无需 LLM）：向量得分分布、关键词覆盖率、来源多样性。
2. 将信号 + 检索片段一起送入 LLM，由 LLM 做综合判断。
3. 解析 LLM 返回的 JSON，填充 :class:`EvaluationResult`。

LLM 调用方式与 ``model_client.api.llm.LLMClient`` 保持一致（不依赖 FlexRAG）。
通过 :meth:`LLMEvaluator.as_tool` 可将评估能力直接暴露为 Parlant SDK 工具。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Annotated, Any, Optional

from rag_module.retrieval.types import Chunk, RetrievalResult
from model_client.api.llm import LLMClient, LLMConfig


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """评估器的输出。"""
    verdict: str          # "qualified" | "insufficient" | "no_coverage"
    relevance: float      # 相关性得分 0~1
    coverage: float       # 覆盖度得分 0~1
    confidence: float     # 综合置信度 0~1
    conclusion: str       # 人类可读的结论
    action_hint: str      # 给 Agent 的后续行动提示
    signals: Optional["RetrievalSignals"] = None


@dataclass
class RetrievalSignals:
    """评估前预计算的量化信号（内部使用）。"""
    avg_score: float
    max_score: float
    min_score: float
    keyword_coverage: float
    source_diversity: int
    chunk_count: int
    missing_keywords: list[str]


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "与", "或", "吗", "呢", "吧",
    "一个", "这个", "那个", "什么", "怎么", "为什么", "如何", "请问",
    "what", "how", "why", "is", "the", "a", "an", "of", "in", "to", "do",
    "does", "did", "can", "could", "would", "should", "will",
}


def _compute_signals(question: str, retrieval: RetrievalResult) -> RetrievalSignals:
    chunks = retrieval.chunks
    scores = [c.score for c in chunks]
    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    max_score = round(max(scores), 4) if scores else 0.0
    min_score = round(min(scores), 4) if scores else 0.0

    question_words = {w for w in question.lower().split() if w and w not in _STOPWORDS and len(w) > 1}
    all_content = " ".join(c.content.lower() for c in chunks)
    covered = question_words & set(all_content.split())
    keyword_coverage = round(len(covered) / len(question_words), 3) if question_words else 0.0

    return RetrievalSignals(
        avg_score=avg_score,
        max_score=max_score,
        min_score=min_score,
        keyword_coverage=keyword_coverage,
        source_diversity=len({c.source for c in chunks}),
        chunk_count=len(chunks),
        missing_keywords=sorted(question_words - covered),
    )


def _build_prompt(question: str, retrieval: RetrievalResult, signals: RetrievalSignals) -> str:
    chunks_text = "\n\n".join(
        f"[片段 {i + 1}]  来源：{c.source}  相似度得分：{c.score:.4f}\n{c.content}"
        for i, c in enumerate(retrieval.chunks)
    )
    missing_kw_str = "、".join(signals.missing_keywords[:8]) if signals.missing_keywords else "无"

    return f"""你是一名专业的 RAG 检索质量评估专家，需要综合多种信息对检索结果作出客观评判。

═══════════════════════════════════════════════
用户问题
═══════════════════════════════════════════════
{question}

═══════════════════════════════════════════════
量化信号（预先计算，供参考）
═══════════════════════════════════════════════
• 片段总数：{signals.chunk_count}
• 向量相似度 — 最高：{signals.max_score:.4f}｜平均：{signals.avg_score:.4f}｜最低：{signals.min_score:.4f}
• 关键词覆盖率：{signals.keyword_coverage:.0%}（未覆盖词：{missing_kw_str}）
• 来源多样性：{signals.source_diversity} 个不同文件

═══════════════════════════════════════════════
检索到的内容片段
═══════════════════════════════════════════════
{chunks_text}

═══════════════════════════════════════════════
评估要求
═══════════════════════════════════════════════
请从以下四个维度综合评判，并给出最终裁决：

1. 语义相关性（relevance）：片段内容与问题在语义上的匹配程度，0~1。
2. 信息覆盖度（coverage）：片段能覆盖问题所有方面的程度，0~1。
3. 综合置信度（confidence）：基于以上片段能否给出可靠答案，0~1。
4. 裁决（verdict）：
   - "qualified"    → 信息足够，可直接据此作答
   - "insufficient" → 有部分相关信息，但不完整
   - "no_coverage"  → 几乎没有覆盖，无法作答

请严格返回如下 JSON，不要包含任何额外文字：
{{
  "relevance": <0到1的浮点数>,
  "coverage": <0到1的浮点数>,
  "confidence": <0到1的浮点数>,
  "verdict": "qualified" | "insufficient" | "no_coverage",
  "conclusion": "<一句话中文结论，说明判断依据>",
  "action_hint": "<给 AI 助手的后续操作建议>"
}}"""


def _parse_json(raw: str) -> dict:
    """从 LLM 原始输出中解析 JSON，容错处理 Markdown 代码块等格式。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match2 = re.search(r"\{[\s\S]*\}", raw)
    if match2:
        try:
            return json.loads(match2.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ---------------------------------------------------------------------------
# LLMEvaluator
# ---------------------------------------------------------------------------

class LLMEvaluator:
    """
    多信号融合 + LLM 综合评估器。

    LLM 调用通过 ``model_client.api.llm.LLMClient`` 完成，保持现有 API 调用方式。
    通过 :meth:`as_tool` 可将评估能力暴露为 Parlant SDK 工具，与
    :meth:`~rag_module.retrieval.parlant.ParlantRetriever.as_tool` 配合使用。

    Usage::

        evaluator = LLMEvaluator()                        # 从 .env 自动加载
        result = await evaluator.evaluate(question, retrieval_result)

    Usage（Parlant 工具）::

        tool = evaluator.as_tool()
        await agent.create_guideline(
            condition="rag_search 工具已返回检索结果",
            action="立即调用 evaluate_retrieval 工具评估本次检索结果。",
            tools=[tool],
        )
    """

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self._client = LLMClient(config=config)

    async def evaluate(self, question: str, retrieval: RetrievalResult) -> EvaluationResult:
        """评估检索结果质量。"""
        if not retrieval.chunks:
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0, coverage=0.0, confidence=0.0,
                conclusion="知识库中未检索到任何相关内容。",
                action_hint="告知用户该问题超出当前知识库的覆盖范围。",
                signals=RetrievalSignals(0.0, 0.0, 0.0, 0.0, 0, 0, []),
            )

        signals = _compute_signals(question, retrieval)
        prompt = _build_prompt(question, retrieval, signals)
        raw = await self._client.chat_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        result = _parse_json(raw)

        return EvaluationResult(
            verdict=result.get("verdict", "no_coverage"),
            relevance=float(result.get("relevance", 0.0)),
            coverage=float(result.get("coverage", 0.0)),
            confidence=float(result.get("confidence", 0.0)),
            conclusion=result.get("conclusion", "LLM 未返回结论。"),
            action_hint=result.get("action_hint", ""),
            signals=signals,
        )

    def as_tool(self) -> Any:
        """
        生成 Parlant ``evaluate_retrieval`` 工具函数。

        接收 ``rag_search`` 工具返回的 ``retrieved_chunks`` JSON，
        调用 :meth:`evaluate` 进行 LLM 综合评估，并将结果以结构化形式返回。

        Returns:
            ``@p.tool`` 装饰的 ``evaluate_retrieval`` 异步函数。

        Raises:
            ImportError: 若未安装 ``parlant`` SDK。
        """
        try:
            import parlant.sdk as p
        except ImportError:
            raise ImportError(
                "LLMEvaluator.as_tool() 需要安装 parlant SDK：\n"
                "pip install parlant"
            )

        evaluator = self

        @p.tool
        async def evaluate_retrieval(
            context: p.ToolContext,
            user_question: Annotated[
                str,
                p.ToolParameterOptions(description="用户的原始问题"),
            ],
            retrieved_chunks_json: Annotated[
                str,
                p.ToolParameterOptions(
                    description=(
                        "rag_search 返回的 retrieved_chunks 字段，"
                        "序列化为 JSON 字符串"
                    )
                ),
            ],
        ) -> p.ToolResult:
            """对 RAG 检索结果进行多信号 LLM 综合评估（相关性、覆盖度、置信度）"""
            try:
                raw_chunks = json.loads(retrieved_chunks_json)
            except (json.JSONDecodeError, TypeError):
                raw_chunks = []

            chunks = [
                Chunk(
                    chunk_id=c.get("chunk_id", ""),
                    content=c.get("content", ""),
                    source=c.get("source", ""),
                    score=float(c.get("score", 0.0)),
                )
                for c in raw_chunks
                if isinstance(c, dict)
            ]
            retrieval = RetrievalResult(
                query=user_question,
                chunks=chunks,
                status="success" if chunks else "no_results",
            )

            eval_result = await evaluator.evaluate(
                question=user_question,
                retrieval=retrieval,
            )

            signals_data: dict = {}
            if eval_result.signals is not None:
                s = eval_result.signals
                signals_data = {
                    "avg_score": s.avg_score,
                    "max_score": s.max_score,
                    "min_score": s.min_score,
                    "keyword_coverage": s.keyword_coverage,
                    "source_diversity": s.source_diversity,
                    "missing_keywords": s.missing_keywords,
                }

            return p.ToolResult(
                data={
                    "verdict": eval_result.verdict,
                    "scores": {
                        "relevance": eval_result.relevance,
                        "coverage": eval_result.coverage,
                        "confidence": eval_result.confidence,
                    },
                    "conclusion": eval_result.conclusion,
                    "action_hint": eval_result.action_hint,
                    "signals": signals_data,
                }
            )

        return evaluate_retrieval
