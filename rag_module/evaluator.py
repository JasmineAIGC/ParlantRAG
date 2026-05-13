"""
LLMEvaluator — 多信号融合 + LLM 综合评估器。

评估流程
--------
1. 本地预计算量化信号（无需 LLM）：向量得分分布、关键词覆盖率、来源多样性。
2. 将信号 + 检索片段一起送入 LLM，由 LLM 做综合判断。
3. 解析 LLM 返回的 JSON，填充 :class:`EvaluationResult`。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from rag_module.retrieval.types import RetrievalResult
from rag_module.llm_client import LLMClient, LLMConfig


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

    Usage::

        evaluator = LLMEvaluator()                        # 从 .env 自动加载
        result = await evaluator.evaluate(question, retrieval_result)
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
