"""
内置评价器实现。

提供两个开箱即用的实现：
  - RuleBasedEvaluator：基于关键词统计规则，无需 LLM，速度快、成本低
  - LLMEvaluator：调用 LLM 进行语义级别评价，准确度更高

宿主系统也可以完全忽略这个文件，自己实现 BaseEvaluator。
"""
from __future__ import annotations

from .base import BaseEvaluator, EvaluationResult, RetrievalResult

# 评价结论阈值（可通过子类覆盖调整）
QUALIFIED_THRESHOLD = 0.45
INSUFFICIENT_THRESHOLD = 0.20

STOPWORDS = {
    "的", "了", "是", "在", "我", "有", "和", "与", "或", "吗", "呢", "吧",
    "what", "how", "why", "is", "the", "a", "an", "of", "in", "to", "do",
}


class RuleBasedEvaluator(BaseEvaluator):
    """
    规则评价器（开箱即用，无需 LLM）。

    基于关键词覆盖率和相关度分数计算三个维度的得分：
    相关性（relevance）、覆盖度（coverage）、置信度（confidence）。

    适合：对速度和成本敏感、评价精度要求不极高的场景。
    """

    def __init__(
        self,
        qualified_threshold: float = QUALIFIED_THRESHOLD,
        insufficient_threshold: float = INSUFFICIENT_THRESHOLD,
    ) -> None:
        self._qualified_threshold = qualified_threshold
        self._insufficient_threshold = insufficient_threshold

    async def evaluate(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> EvaluationResult:

        # ── 无结果直接返回 ─────────────────────────────────
        if not retrieval.chunks:
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0,
                coverage=0.0,
                confidence=0.0,
                conclusion="知识库中未找到与该问题相关的任何内容。",
                action_hint="告知用户该问题超出当前知识库的覆盖范围。",
            )

        question_words = set(question.lower().split()) - STOPWORDS

        # ── 维度一：相关性 ──────────────────────────────────
        chunk_relevances: list[float] = []
        for chunk in retrieval.chunks:
            chunk_words = set(chunk.content.lower().split())
            overlap = len(question_words & chunk_words)
            keyword_ratio = overlap / len(question_words) if question_words else 0.0
            combined = round(keyword_ratio * 0.5 + chunk.score * 0.5, 3)
            chunk_relevances.append(combined)

        avg_relevance = round(sum(chunk_relevances) / len(chunk_relevances), 3)

        # ── 维度二：覆盖度 ──────────────────────────────────
        all_content = " ".join(c.content.lower() for c in retrieval.chunks)
        all_words = set(all_content.split())
        covered = question_words & all_words
        coverage = round(len(covered) / len(question_words), 3) if question_words else 0.0
        coverage_gaps = list(question_words - all_words)

        # ── 维度三：置信度（加权综合） ─────────────────────
        valid_count = sum(1 for r in chunk_relevances if r >= 0.15)
        valid_ratio = valid_count / len(retrieval.chunks)
        confidence = round(
            avg_relevance * 0.45 + coverage * 0.35 + valid_ratio * 0.20, 3
        )

        # ── 评价结论 ────────────────────────────────────────
        if confidence >= self._qualified_threshold:
            verdict = "qualified"
            conclusion = (
                f"检索结果质量合格。{len(retrieval.chunks)} 个片段中 "
                f"{valid_count} 个相关，关键词覆盖率 {coverage:.0%}。"
            )
            action_hint = "可直接基于检索内容作答，并注明信息来源。"

        elif confidence >= self._insufficient_threshold:
            verdict = "insufficient"
            conclusion = (
                f"检索结果相关性不足（置信度 {confidence:.2f}），"
                f"关键词覆盖率仅 {coverage:.0%}。"
                + (f"未覆盖关键词：{', '.join(coverage_gaps[:5])}" if coverage_gaps else "")
            )
            action_hint = "建议用更具体的关键词重新提问，或告知用户信息覆盖有限。"

        else:
            verdict = "no_coverage"
            conclusion = (
                f"知识库对该问题几乎没有覆盖（置信度 {confidence:.2f}），"
                f"无法给出可靠回答。"
            )
            action_hint = "告知用户该问题超出知识库范围，建议联系人工支持。"

        return EvaluationResult(
            verdict=verdict,
            relevance=avg_relevance,
            coverage=coverage,
            confidence=confidence,
            conclusion=conclusion,
            action_hint=action_hint,
            details={
                "chunk_relevances": chunk_relevances,
                "coverage_gaps": coverage_gaps,
                "valid_chunk_count": valid_count,
            },
        )


class LLMEvaluator(BaseEvaluator):
    """
    LLM 评价器（需要 OpenAI 兼容 API）。

    使用 LLM 进行语义级别的相关性判断，准确度更高，但有额外 LLM 调用成本。

    适合：对评价质量要求高、允许额外 LLM 调用成本的场景。

    Usage::

        evaluator = LLMEvaluator(
            api_key="sk-...",
            model="gpt-4o-mini",
        )
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url

    async def evaluate(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> EvaluationResult:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("LLMEvaluator 需要安装：pip install openai")

        if not retrieval.chunks:
            return EvaluationResult(
                verdict="no_coverage",
                relevance=0.0,
                coverage=0.0,
                confidence=0.0,
                conclusion="未检索到任何内容。",
                action_hint="告知用户该问题超出知识库范围。",
            )

        client = AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)
        chunks_text = "\n\n".join(
            f"[片段{i + 1}] 来源：{c.source}\n{c.content}"
            for i, c in enumerate(retrieval.chunks)
        )

        prompt = f"""你是一个 RAG 检索质量评估专家。

用户问题：{question}

检索到的内容片段：
{chunks_text}

请评估这批检索内容的质量，返回严格的 JSON 格式：
{{
  "relevance": 0到1的小数（片段与问题的平均相关程度）,
  "coverage": 0到1的小数（片段能覆盖问题多少方面）,
  "confidence": 0到1的小数（基于这批内容能否可靠作答）,
  "verdict": "qualified"（可靠作答）或 "insufficient"（信息不足）或 "no_coverage"（无覆盖）,
  "conclusion": "一句话评价结论",
  "action_hint": "给AI助手的建议"
}}

只返回 JSON，不要其他文字。"""

        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        import json
        result = json.loads(response.choices[0].message.content)

        return EvaluationResult(
            verdict=result.get("verdict", "no_coverage"),
            relevance=float(result.get("relevance", 0.0)),
            coverage=float(result.get("coverage", 0.0)),
            confidence=float(result.get("confidence", 0.0)),
            conclusion=result.get("conclusion", ""),
            action_hint=result.get("action_hint", ""),
        )
